import logging
import json
from nano_llm import Plugin
from nano_llm.plugins.robotics.ros_connector import NodeType

class MotionQuery(Plugin):
    def __init__(self, **kwargs):
        
        super().__init__(outputs=['json_in'], threaded=False, **kwargs)
        logging.info("Motion Query plugin initialized")

        self.add_parameter('robot_namespace', type=str, default="")

        self.add_tool(self.move_finite)

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
            "msg": twist_msg
        }
        print("do we get here")

        self.output(json.dumps(json_dict))

        return "Moved a finite distance"
        


        

