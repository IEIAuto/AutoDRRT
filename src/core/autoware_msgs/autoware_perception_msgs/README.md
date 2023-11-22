# autoware_perception_msgs

## TrafficLightElement.msg

This is the element of traffic lights such as red, amber, green, and turn arrow. The elements are based on Vienna Convention on Road Signs and Signals.

## TrafficLight.msg

For each direction of an intersection, there are typically multiple traffic lights installed for visibility.
This message represents the traffic light as equipment and is used by components such as perception.

## TrafficSignal.msg

For each direction of an intersection, there is one state of traffic signal regardless of the number of equipment.
This message represents the traffic signal as a concept and is used by components such as planning.

## TrafficLightArray.msg

This is a plural type of TrafficLight.msg.

## TrafficSignalArray.msg

This is a plural type of TrafficSignal.msg.
