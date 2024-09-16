import logging
from nano_llm import Plugin
from ros_connector import NodeType

class MotionQuery(Plugin):
    def __init__(self, **kwargs):
        
        super().__init__(outputs=['json_in'], threaded=False, **kwargs)
        logging.info("Motion Query plugin initialized")

        self.add_parameter('robot_namespace', type=str, default="")



    def move_finite(self):
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
        ## TESTING


        

