from enum import Enum

from autoware_auto_vehicle_msgs.msg import TurnIndicatorsReport
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy


class TurnIndicatorsReport_Constants(Enum):
    DISABLE = 1
    ENABLE_LEFT = 2
    ENABLE_RIGHT = 3


class SubscriberTurnIndicatorsReport(Node):
    def __init__(self):
        super().__init__("turn_indicators_report_subscriber")

        self.declare_parameter("qos_depth", 10)
        qos_depth = self.get_parameter("qos_depth").value

        QOS_RKL10V = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.topic = "/vehicle/status/turn_indicators_status"
        self.subscription_ = self.create_subscription(
            TurnIndicatorsReport, self.topic, self.get_status, QOS_RKL10V
        )

        self.received = []

    def get_status(self, msg):
        self.received.append(msg)


def main(args=None):
    rclpy.init(args=args)

    node = SubscriberTurnIndicatorsReport()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
