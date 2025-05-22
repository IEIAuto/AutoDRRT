import rclpy
import os
from rclpy.node import Node

from std_msgs.msg import String
from autoware_adapi_v1_msgs.msg import RouteState
from geometry_msgs.msg import PoseStamped
from autoware_auto_vehicle_msgs.msg import Engage


class MinimalSubscriber(Node):

    def __init__(self):

        super().__init__('minimal_subscriber')
        self.sub_state = self.create_subscription(RouteState,'/planning/mission_planning/route_state',self.state_callback,10)
        self.sub_position = self.create_subscription(PoseStamped,'/localization/pose_twist_fusion_filter/pose',self.position_callback,10)

        self.tmp_pose = ""

        self.pub_des = self.create_publisher(PoseStamped, '/planning/mission_planning/goal', 10)
        self.pub_engage = self.create_publisher(Engage, '/autoware/engage', 10)
        
        self.eng = Engage()
        # self.eng.stamp  = self.time_transform()
        self.eng.stamp.sec = 0
        self.eng.stamp.nanosec = 0
        self.eng.engage = True

    def time_transform(self):
        t = self.get_clock().now()
        return t.to_msg()
        
 
    def state_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.state)
        # self.get_logger().info('pose: "%s"' % self.tmp)
        if msg.state==3:
            self.get_logger().info('final pose: "%s"' % self.tmp_pose)
            self.get_logger().info('final state: "%s"' % msg.state)
                    
            print(self.tmp_pose)

            if abs(self.tmp_pose.position.x - 81688.8203125) > abs(self.tmp_pose.position.x - 81789.5390625):
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp  = self.time_transform()
                pose.pose.position.x = 81688.8203125
                pose.pose.position.y = 50288.9921875
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = -0.5965673853859907
                pose.pose.orientation.w = 0.8025629911064445

                self.pub_des.publish(pose)
                os.system("/bin/bash /home/orin/autoware_awsim/2d_pose.sh 1")

            else:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp  = self.time_transform()
                pose.pose.position.x = 81789.5390625
                pose.pose.position.y = 50371.921875
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.7800418122827314
                pose.pose.orientation.w = 0.6257273935913882
                
                self.pub_des.publish(pose)
                os.system("/bin/bash /home/orin/autoware_awsim/2d_pose.sh 1")
                
    def position_callback(self, msg):
        #  self.get_logger().info('state: "%s"' % msg.pose.position.x)
         self.tmp_pose = msg.pose
     
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


