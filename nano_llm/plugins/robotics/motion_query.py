import logging
import json
from nano_llm import Plugin
from nano_llm.plugins.robotics.ros_connector import NodeType, ROSLog, LogLevel

class MotionQuery(Plugin):
    def __init__(self, **kwargs):
        
        super().__init__(outputs=['json_in'], threaded=False, **kwargs)
        logging.info("Motion Query plugin initialized")

        self.add_parameter('robot_namespace', type=str, default="")

        self.add_tool(self.move_finite)
        self.add_tool(self.topic_subscriber)

    def move_finite(self, distance: float, speed: float, direction: int):
        """
        Move forwards or backwards by a certain distance on the map at a certain velocity. Use this tool when the user asks you to move
        forward or backward by a certain distance. The distance unit will always be in meters even despite the user specifying otherwise.
        The direction of movement will always be forward or backward with respect to the robot's current orientation.

        Args:
            distance: a float representing the distance in meters to travel on the map
            speed: a float in m/s
            direction: {-1, 1} where -1 is passed should the user specify backwards as the direction of travel
                               and 1 is passed should the user specify forwards as the direction of travel

        Returns:
            A json dict (format and contents unknown at this time)
        """
        robot_namespace = self.robot_namespace
        twist_msg = {
            "linear": {
                "x": direction * speed,
                "y": 0.0,
                "z": 0.0
            },
            "angular": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            }
        }
        json_dict = {
            "node_type": NodeType('publisher'),
            "msg_type": 'geometry_msgs/msg/Twist',
            "name": f"{robot_namespace}/cmd_vel",
            "timer_period": 0.05,
            "timer_duration": abs(distance / speed),
            "msg": twist_msg,
            # "ros_log": ROSLog(name="cmd_vel_log", level=LogLevel('INFO'), msg="created publisher for cmd_vel")
        }

        self.output(json.dumps(json_dict))

        return "Moved a finite distance"


    def topic_subscriber(self, topic_name: str, message_type: str):
        """
        Use this tool whenever you are asked to subscribe to a topic. The user should also tell you what message type they would
        like to subscribe to on said topic. If they do not, do not call this tool and instead prompt the user to input the message type
        and to confirm the topic they wish to subscribe to.

        Args:
            topic_name: the topic that the user wishes to subscribe to
            message_type: the message type that the user wishes to subscribe to over topic_name. 

        Returns:
            A basic string that is not related to the information being funneled across the topic.
        """

        robot_namespace = self.robot_namespace

        json_dict = {
            "node_type": NodeType('subscriber'),
            "msg_type": 'geometry_msgs/msg/Twist',
            "name": f"{robot_namespace}/{topic_name}",
            "timer_period": 0.05,
            "msg": {}
        }
        
        self.output(json.dumps(json_dict))

        return "test"


        

