#!/usr/bin/env python3
import logging
from functools import partial
from nano_llm import Plugin
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
import rclpy.logging
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rosidl_runtime_py import set_message, convert
import importlib
from queue import Queue
from enum import Enum
from typing import Optional, Annotated
from pydantic import BaseModel, Field, ValidationError
import threading

########################################################
##### Pydantic schemas for ROS2 message validation #####
########################################################

class LogLevel(str, Enum):
    """
    Enum for ROS2 log levels
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"

class NodeType(str, Enum):
    """
    Enum for ROS2 object types
    """
    PUBLISHER = "publisher"
    SUBSCRIBER = "subscriber"
    SERVICE_CLIENT = "service_client"
    ACTION_CLIENT = "action_client"

class ROSLog(BaseModel):
    """

    Pydantic schema for ROS2 log messages
    """
    name: Optional[str] = Field(description = "the name of the logger")
    level: Optional[LogLevel] = Field(description = "the log level, e.g. 'LogLevel.INFO'", default = LogLevel.INFO) 
    msg: str = Field(description = "the log message emitted when the logger is created")

class ROSMessage(BaseModel):
    """
    Pydantic schema for ROS2 topic, service client request, and action client goal messages
    """
    node_type: NodeType = Field(description = "the type of ROS2 node either 'publisher', 'subscriber', 'service_client', or 'action_client'")
    msg_type: str = Field(description = "the type of ROS2 message, service, or action, e.g. 'std_msgs/msg/String'")
    name: str = Field(description = "the name of the ROS2 topic, service, or client, e.g. 'chatter'")
    timer_period: Annotated[float, Field(description = "the period of the timer, ignored if type is not 'publisher'", default=0, ge=0.0)]
    msg: dict = Field(description = "the message payload for the topic, service request/response, or action goal/result")
    ros_log: Optional[ROSLog] = Field(description = "optional ros logging message to be emitted when logger is created", default='')


########################################################
##### ROS2Connector plugin for converting messages #####
########################################################

class ROS2Connector(Plugin, Node):
    """super()
    Plugin to take in well-formed JSON messages and convert them to ROS2 messages and vice versa.
    It dynamically creates ROS2 publishers/subscribers, service clients, and action clients based 
    on the JSON message.

    Inputs: (dict) -- JSON dictionary to be converted to ROS2 message
            NOTE: - JSON message must be well-formed and follow the ROSMessage schema.
                  - A publisher, subscriber, service client, or action client can be destroyed
                    by send the case-insensitive string 'DESTROY' in the 'msg' attribute of the input.
                  - An action goal can be canceled by sending the case-insensitive string 'CANCEL' in 
                    the 'msg' attribute of the input.

    Outputs: (str) -- JSON message converted from ROS2 message
    """

    def __init__(self, **kwargs):
        """
        Take in JSON message and parse to create ROS2 publishers, subscribers,
        service clients, and action clients. Publish on and subscribe to ROS2 topics
        and send service requests and action goals. Output JSON messages constructed
        from messages and responses.

        Args:
            init_args (list): list of arguments to pass to the ROS2 node

        """
        Plugin.__init__(self, inputs=['json_in'], outputs=['json_out'],**kwargs)

        # Initialize rclpy
        rclpy.init(args=kwargs.get('init_args', []))
        # Initialize node
        self.node = rclpy.create_node('ros2_connector')
        # Create the MultiThreadedExecutor
        self.exec = MultiThreadedExecutor()
        # Add the node to the executor
        self.exec.add_node(self.node)
        self._executor_thread = None
    
        # ^ Where are we spinning this node? It appears that it will still be a blocking node unless we
        # explicitly run the executor in a different thread in parallel with main script. 
        # Understand that all of the functions defined below are considered callbacks, and adding this node
        # to a multi-threaded executor allows these callbacks to execute in parallel (in separate threads)


        self.node.get_logger().info("ROS2Connector plugin started")
        self.node.get_logger().info("ROS2 node created")

        # Initialize subnode dictionaries
        self.pubs = {}
        self.subs = {}
        self.service_clients = {}
        self.action_clients = {}
        # Initialize logger, timer, callback group, and goal handle dictionaries
        self.loggers = {}
        self.timers_dict = {}
        self.callback_groups = {}
        self.goal_handles = {}

        self.log_levels = {
            "DEBUG": rclpy.logging.LoggingSeverity.DEBUG,
            "INFO": rclpy.logging.LoggingSeverity.INFO,
            "WARN": rclpy.logging.LoggingSeverity.WARN,
            "ERROR": rclpy.logging.LoggingSeverity.ERROR,
            "FATAL": rclpy.logging.LoggingSeverity.FATAL
        }

    def get_ros_msg_from_json(self, json_msg: dict) -> ROSMessage:
        """
        Validate JSON input and cast to ROSMessage.
        """
        try:
            ros_msg = ROSMessage.model_validate_json(json_msg)
        except ValidationError as e:
            print(f"Invalid ROS2 message: {e}")
            self.get_logger().error(f"Invalid ROS2 message sent by agent: {e}")
            return False
        return ros_msg

    def get_ros_message_type(self, ros_msg: ROSMessage) -> tuple:
        """
        Get the ROS2 message type from the JSON message and import the class.
        """
        msg_type_str = ros_msg.msg_type
        package, msg_dir, msg_type = msg_type_str.split('/')
        # get the message class and import module and class
        msg_module = importlib.import_module(f'{package}.{msg_dir}')
        msg_class = getattr(msg_module, msg_type)
        globals()[msg_type] = msg_class
        return msg_type, msg_class

           
    def create_publisher(self, msg: ROSMessage, msg_class) -> bool:
        """
        Create a ROS2 publisher.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        self.create_logger(msg)
        self.callback_group[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class = self.get_ros_message_type(msg)
        pub_msg = self.json_to_ros_msg(msg, msg_class)
        try:
            if not self.pubs.get(msg.name):
                self.pubs[msg.name] = self.node.create_publisher(msg_class, 
                                                                       msg.name, 
                                                                       10, 
                                                                       callback_group=self.callback_groups[msg.name])
                if msg.timer_period != 0:
                    self.timers_dict[msg.name] = self.node.create_timer(msg.timer_period, 
                                                                   partial(self.timer_callback, msg=pub_msg, 
                                                                   callback_group=self.callback_groups[msg.name]))
            return True
        except Exception as e:
            self.publish_log(self.loggers[msg.ros_log.name], f"Failed to create publisher: {e}", log_level=LogLevel.ERROR)
            return False
        
    def timer_callback(self, msg: ROSMessage) -> None:
        """
        Timer callback function for ROS2 publishers.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        self.publish_ros_message(msg)
        self.publish_log(self.loggers[msg.ros_log.name], f"Published message to topic: {msg.name}", log_level=LogLevel.INFO)
    
    def create_subscriber(self, msg: ROSMessage) -> bool:
        """
        Create a ROS2 subscriber.
        """
        assert(msg.node_type == NodeType.SUBSCRIBER)
        self.create_logger(msg)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class = self.get_ros_message_type(msg)
        try:
            self.subs[msg.name] = self.node.create_subscription(msg_class, 
                                                                    msg.name, 
                                                                    partial(self.subscriber_callback, ros_msg=msg), 
                                                                    10,
                                                                    callback_group=self.callback_groups[msg.name])
            return True
        except Exception as e:
            self.publish_log(self.loggers[msg.ros_log.name], f"Failed to create subscriber: {e}", log_level=LogLevel.ERROR)
            return False

    def subscriber_callback(self, msg, ros_msg: ROSMessage) -> None:
        """
        Callback function for ROS2 subscribers. Replace payload of original message with message
        received by subscriber and JSON dump to output.
        """
        self.publish_log(self.loggers[ros_msg.ros_log.name], f"Received message from topic: {ros_msg.name}", log_level=LogLevel.INFO)
        json_msg = self.ros_msg_to_json(msg)
        # replace payload of original message with received message
        ros_msg.msg = json_msg
        # convert ROS2Message to JSON dict
        out_msg = ros_msg.model_dump()
        # put received message in output queue as JSON dict
        self.output(out_msg, 0)

    def create_service_client(self, msg: ROSMessage) -> bool:
        """
        Create a ROS2 service client.
        """
        assert(msg.node_type == NodeType.SERVICE_CLIENT)
        self.create_logger(msg)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class = self.get_ros_message_type(msg)
        self.service_clients[msg.name] = self.node.create_client(msg_class, msg.name)
        while not self.service_clients[msg.name].wait_for_service(timeout_sec=1.0):
            self.publish_log(self.loggers[msg.ros_log.name], f"service {msg.name} not available, waiting again...", log_level=LogLevel.INFO)
        return True

    def create_action_client(self, msg: ROSMessage) -> bool:
        """
        Create a ROS2 action client.
        """
        assert(msg.node_type == NodeType.ACTION_CLIENT)
        self.create_logger(msg)
        self.callback_groups[msg.name] = MutuallyExclusiveCallbackGroup()
        _, msg_class = self.get_ros_message_type(msg)
        self.action_clients[msg.name] = ActionClient(self.node, msg_class, msg.name)
        while not self.action_clients[msg.name].wait_for_server(timeout_sec=1.0):
            self.publish_log(self.loggers[msg.ros_log.name], f"action server {msg.name} not available, waiting again...", log_level=LogLevel.INFO)
        return True

    def create_logger(self, msg: ROSMessage):
        """
        Create a ROS2 logger.
        """
        if not msg.ros_log.name:
            msg.ros_log.name = f"{msg.name}_log"
        try:
            log_level = self.log_levels.get(msg.ros_log.level.upper(), rclpy.logging.LoggingSeverity.INFO)
            ros_logger = self.node.get_logger().get_child(msg.ros_log.name)
            ros_logger.set_level(log_level)
            ros_logger.log(log_level, msg.ros_log.msg)
            self.loggers[msg.ros_log.name] = ros_logger
        except Exception as e:
            self.node.get_logger().error(f"Failed to create logger: {e}")
            return False
        return True

    def publish_log(self, logger: RcutilsLogger, msg: str, log_level: LogLevel=LogLevel.INFO) -> bool:
        """
        Publish a ROS2 log message.
        """
        try:
            match log_level:
                case LogLevel.DEBUG:
                    logger.debug(msg)
                case LogLevel.INFO:
                    logger.info(msg)
                case LogLevel.WARN:
                    logger.warn(msg)
                case LogLevel.ERROR:
                    logger.error(msg)
                case LogLevel.FATAL:
                    logger.fatal(msg)
                case _:
                    logger.info(msg)
        except Exception as e:
            print(f"Failed to publish log message: {e}")
            self.node.get_logger().error(f"Failed to publish log message: {e}")
            return False
        return True

    def publish_ros_message(self, msg: ROSMessage) -> bool:
        """
        Publish a ROS2 message to a topic.
        """
        assert(msg.node_type == NodeType.PUBLISHER)
        _, msg_class = self.get_ros_message_type(msg)
        ros_msg = self.json_to_ros_msg(msg, msg_class)
        try:
            publisher = self.pubs.get(msg.name)
            publisher.publish(ros_msg)
            self.publish_log(self.loggers[msg.ros_log.name], f"Published message to topic: {msg.name}", log_level=LogLevel.INFO)
            return True
        except Exception as e:
            self.publish_log(self.loggers[msg.ros_log.name], f"Failed to publish message: {e}", log_level=LogLevel.ERROR)
            return False
        
    def send_service_request_async(self, msg: ROSMessage) -> bool:
        """
        Send a service request to a ROS2 service client, await response, and output response.
        """
        assert(msg.node_type == NodeType.SERVICE_CLIENT)
        _, msg_class = self.get_ros_message_type(msg)
        try:
            request_msg = self.json_to_ros_msg(msg, msg_class.Request)
            future = self.service_clients[msg.name].call_async(request_msg)
            result_msg = future.result()
            self.publish_log(self.loggers[msg.ros_log.name], f"Received response from service: {msg.name}", log_level=LogLevel.INFO)
            result_json = self.ros_msg_to_json(result_msg)
            # replace payload of original message with received message
            msg.msg = result_json
            # convert ROS2Message to JSON dict
            out_msg = msg.model_dump()
            # put received message in output queue as JSON dict
            self.output(out_msg, 0)
            return True
        except Exception as e:
            self.publish_log(self.loggers[msg.ros_log.name], f"Failed to send service request: {e}", log_level=LogLevel.ERROR)

    def send_action_goal(self, msg: ROSMessage) -> bool:
        assert(msg.node_type == NodeType.ACTION_CLIENT)
        action_client = self.action_clients[msg.name]
        _, msg_class = self.get_ros_message_type(msg)
        
        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.publish_log(self.loggers[msg.ros_log.name], f"Goal rejected by action server: {msg.name}", log_level=LogLevel.ERROR)
                return
            self.publish_log(self.loggers[msg.ros_log.name], f"Goal accepted by action server: {msg.name}", log_level=LogLevel.INFO)
            _get_result_future = goal_handle.get_result_async()
            _get_result_future.add_done_callback(get_result_callback)

        def feedback_callback(feedback):
            feedback = feedback.feedback.sequence
            self.publish_log(self.loggers[msg.ros_log.name], 
                             'Received feedback: {0}'.format(feedback), 
                             log_level=LogLevel.INFO)
            feedback_json = self.ros_msg_to_json(feedback)
            # replace payload in copy of original message with received message (will need original message again for result callback)
            ros_msg = msg
            ros_msg.msg = feedback_json
            # convert ROS2Message to JSON dict
            out_msg = ros_msg.model_dump()
            # put received message in output queue as JSON dict
            self.output(out_msg, 0)

        def get_result_callback(future):
            result = future.result().result
            status = future.result().status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.publish_log(self.loggers[msg.ros_log.name], f"Goal succeeded! Result: {result.sequence}", log_level=LogLevel.INFO)
                result_json = self.ros_msg_to_json(result.sequence)
                # replace payload in copy of original message with received message
                ros_msg = msg
                ros_msg.msg = result_json
                # convert ROS2Message to JSON dict
                out_msg = ros_msg.model_dump()
                # put received message in output queue as JSON dict
                self.output(out_msg, 0)
            else:
                self.publish_log(self.loggers[msg.ros_log.name], f"Goal failed with status: {status}", log_level=LogLevel.ERROR)

        def send_goal():
            self.publish_log(self.loggers[msg.ros_log.name], f"Sending goal request to action server: {msg.name}", log_level=LogLevel.INFO)
            action_client.wait_for_server()
            goal_msg_json = msg.msg
            goal_msg = self.json_to_ros_msg(goal_msg_json, msg_class.Goal)
            self.publish_log(self.loggers[msg.ros_log.name], f"Sending goal request...", log_level=LogLevel.INFO)
            _send_goal_future = action_client.send_goal_async(
                                                              goal_msg,
                                                              feedback_callback=feedback_callback)
            _send_goal_future.add_done_callback(goal_response_callback)
            
        # This will return a goal handle that can be used to cancel the goal
        try:
            action_goal_handle = send_goal()
        except Exception as e:
            self.publish_log(self.loggers[msg.ros_log.name], f"Failed to send action goal: {e}", log_level=LogLevel.ERROR)
            return
        return action_goal_handle

    def process(self, input: dict, **kwargs): 
        """
        Create a ROS2 publisher, subscriber, service client, or action client.
        Publish messages to topics, subscribe to topics, send service requests, and send action goals.
        """
        json_msg = input
        msg = self.get_ros_msg_from_json(json_msg)
        msg_type, msg_class = self.get_ros_message_type(msg)
        # Convert JSON message payload to ROS2 message for any publishing
        ros_msg = self.json_to_ros_msg(msg, msg_class)

        match msg_type:

            case NodeType.PUBLISHER:
                # Don't bother creating a publisher if it's already been created and don't create a new one for a 'destroy' message
                if not self.pubs.get(msg.name) and not msg.msg.lower() == 'destroy':
                    self.create_publisher(msg, msg_class)
                    if msg.timer_period == 0:
                        self.publish_ros_message(msg)
                        self.publish_log(self.loggers[msg.ros_log.name], f"Published message to topic: {msg.name}", log_level=LogLevel.INFO)
                elif self.pubs.get(msg.name) and not msg.msg.lower() == 'destroy':
                    if msg.timer_period == 0:
                        if self.timers_dict.get(msg.name):
                            self.timers_dict[msg.name].destroy()
                            del self.timers_dict[msg.name]
                        self.publish_ros_message(msg)
                        self.publish_log(self.loggers[msg.ros_log.name], f"Published message to topic: {msg.name}", log_level=LogLevel.INFO)
                    else:
                        if self.timers_dict.get(msg.name):
                            self.timers_dict[msg.name].destroy()
                            del self.timers_dict[msg.name]
                        self.timers_dict[msg.name] = self.node.create_timer(msg.timer_period, 
                                                                       partial(self.timer_callback, msg=ros_msg, 
                                                                       callback_group=self.callback_groups[msg.name]))
                elif msg.msg.lower() == 'destroy':
                    if self.timers_dict.get(msg.name):
                        self.timers_dict[msg.name].destroy()
                        del self.timers_dict[msg.name]
                    self.pubs[msg.name].destroy()
                    del self.pubs[msg.name]
                    del self.callback_groups[msg.name]
                    del self.loggers[msg.ros_log.name]
                    self.publish_log(self.loggers[msg.ros_log.name], f"Publisher destroyed: {msg.name}", log_level=LogLevel.INFO)

            case NodeType.SUBSCRIBER:
                if not self.subs.get(msg.name):
                    self.create_subscriber(msg, msg_class)
                if msg.msg.lower() == 'destroy':
                    self.subs[msg.name].destroy()
                    del self.subs[msg.name]
                    del self.callback_groups[msg.name]
                    del self.loggers[msg.ros_log.name]
                    self.publish_log(self.loggers[msg.ros_log.name], f"Subscriber destroyed: {msg.name}", log_level=LogLevel.INFO)
            
            case NodeType.SERVICE_CLIENT:
                if not self.service_clients.get(msg.name):
                    self.create_service_client(msg, msg_class)
                    self.send_service_request_async(msg)
                elif msg.msg.lower() == 'destroy':
                    self.service_clients[msg.name].destroy()
                    del self.service_clients[msg.name]
                    del self.callback_groups[msg.name]
                    del self.loggers[msg.ros_log.name]
                    self.publish_log(self.loggers[msg.ros_log.name], f"Service client destroyed: {msg.name}", log_level=LogLevel.INFO)
            
            case NodeType.ACTION_CLIENT:
                # Don't bother creating an action client if it's already been created and don't create one if the message is 'cancel' or 'destroy'
                if not self.action_clients.get(msg.name) and not msg.msg.lower() == 'cancel' and not msg.msg.lower() == 'destroy':
                    self.create_action_client(msg, msg_class)
                    self.goal_handles[msg.name] = self.send_action_goal(msg)
                elif msg.msg.lower() == 'cancel':
                    self.goal_handles[msg.name].cancel_goal()
                    self.publish_log(self.loggers[msg.ros_log.name], f"Goal canceled for action server: {msg.name}", log_level=LogLevel.INFO)
                elif msg.msg.lower() == 'destroy':
                    self.action_clients[msg.name].destroy()
                    del self.action_clients[msg.name]
                    del self.callback_groups[msg.name]
                    del self.loggers[msg.ros_log.name]
                    del self.goal_handles[msg.name]
                    self.publish_log(self.loggers[msg.ros_log.name], f"Action client destroyed: {msg.name}", log_level=LogLevel.INFO)
            
            case _:
                self.node.get_logger().error(f"Invalid ROS2 node type: {msg.node_type}")
    
    # Override the Plugin.run method to spin the ROS2 node
    def run(self):
        """
        Processes the queue forever and automatically run when created with ``threaded=True``
        """
        ### Spin the node############
        # self.executor.spin(self.node)
        self._executor_thread = threading.Thread(target=self.exec.spin)
        self._executor_thread.start()
        #############################

        while not self.stop_flag:
            try:
                self.process_inputs(timeout=0.25)
            except Exception as error:
                logging.error(f"Exception occurred during processing of {self.name}\n\n{traceback.format_exc()}")

        logging.debug(f"{self.name} plugin stopped (thread {self.native_id})")
    
    # Override the Plugin.remove_plugin method to shutdown and remove the ROS2 node
    def destroy(self):
        """
        Stop a plugin thread's running, and unregister it from the global instances.
        """ 
        ### Shut down and Destroy the node #####
        self.exec.shutdown()
        self.node.destroy_node('ros2_connector')
        rclpy.shutdown()
        ########################################

        self.stop()
                
        try:
            Plugin.Instances.remove(self)
        except ValueError:
            logging.warning(f"Plugin {getattr(self, 'name', '')} wasn't in global instances list when being deleted")
        
        plugin.destroy()
        del plugin

    def json_to_ros_msg(self, msg, msg_class):
        """
        Convert JSON message to ROS2 message.
        """
        ros_msg = msg_class()
        msg_dict = msg.msg
        set_message.set_message_fields(ros_msg, msg_dict)
        return ros_msg

    def ros_msg_to_json(self, ros_msg) -> dict:
        """
        Convert ROS2 message to JSON message.
        """
        return convert.message_to_ordered_dict(ros_msg)

    def state_dict(self, **kwargs):
        return {**super().state_dict(**kwargs)}
