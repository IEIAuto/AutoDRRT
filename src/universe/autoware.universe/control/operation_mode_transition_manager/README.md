# operation_mode_transition_manager

## Purpose / Use cases

This module is responsible for managing the different modes of operation for the Autoware system. The possible modes are:

- `Autonomous`: the vehicle is fully controlled by the autonomous driving system
- `Local`: the vehicle is controlled by a physically connected control system such as a joy stick
- `Remote`: the vehicle is controlled by a remote controller
- `Stop`: the vehicle is stopped and there is no active control system.

There is also an `In Transition` state that occurs during each mode transitions. During this state, the transition to the new operator is not yet complete, and the previous operator is still responsible for controlling the system until the transition is complete. Some actions may be restricted during the `In Transition` state, such as sudden braking or steering. (This is restricted by the `vehicle_cmd_gate`).

### Features

- Transit mode between `Autonomous`, `Local`, `Remote` and `Stop` based on the indication command.
- Check whether the each transition is available (safe or not).
- Limit some sudden motion control in `In Transition` mode (this is done with `vehicle_cmd_gate` feature).
- Check whether the transition is completed.

- Transition between the `Autonomous`, `Local`, `Remote`, and `Stop` modes based on the indicated command.
- Determine whether each transition is safe to execute.
- Restrict certain sudden motion controls during the `In Transition` mode (using the `vehicle_cmd_gate` feature).
- Verify that the transition is complete.

## Design

<!-- ## Assumptions / Known limits -->

## Inputs / Outputs / API

### Inputs

For the mode transition:

- /system/operation_mode/change_autoware_control [`tier4_system_msgs/srv/ChangeAutowareControl`]: change operation mode to Autonomous
- /system/operation_mode/change_operation_mode [`tier4_system_msgs/srv/ChangeOperationMode`]: change operation mode

For the transition availability/completion check:

- /control/command/control_cmd [`autoware_auto_control_msgs/msg/AckermannControlCommand`]: vehicle control signal
- /localization/kinematic_state [`nav_msgs/msg/Odometry`]: ego vehicle state
- /planning/scenario_planning/trajectory [`autoware_auto_planning_msgs/msg/Trajectory`]: planning trajectory
- /vehicle/status/control_mode [`autoware_auto_vehicle_msgs/msg/ControlModeReport`]: vehicle control mode (autonomous/manual)
- /control/vehicle_cmd_gate/operation_mode [`autoware_adapi_v1_msgs/msg/OperationModeState`]: the operation mode in the `vehicle_cmd_gate`. (To be removed)

For the backward compatibility (to be removed):

- /api/autoware/get/engage [`autoware_auto_vehicle_msgs/msg/Engage`]
- /control/current_gate_mode [`tier4_control_msgs/msg/GateMode`]
- /control/external_cmd_selector/current_selector_mode [`tier4_control_msgs/msg/ExternalCommandSelectorMode`]

### Outputs

- /system/operation_mode/state [`autoware_adapi_v1_msgs/msg/OperationModeState`]: to inform the current operation mode
- /control/operation_mode_transition_manager/debug_info [`operation_mode_transition_manager/msg/OperationModeTransitionManagerDebug`]: detailed information about the operation mode transition

- /control/gate_mode_cmd [`tier4_control_msgs/msg/GateMode`]: to change the `vehicle_cmd_gate` state to use its features (to be removed)
- /autoware/engage [`autoware_auto_vehicle_msgs/msg/Engage`]:

- /control/control_mode_request [`autoware_auto_vehicle_msgs/srv/ControlModeCommand`]: to change the vehicle control mode (autonomous/manual)
- /control/external_cmd_selector/select_external_command [`tier4_control_msgs/srv/ExternalCommandSelect`]:

## Parameters

| Name                               | Type     | Description                                                                                       | Default value |
| :--------------------------------- | :------- | :------------------------------------------------------------------------------------------------ | :------------ |
| `transition_timeout`               | `double` | If the state transition is not completed within this time, it is considered a transition failure. | 10.0          |
| `frequency_hz`                     | `double` | running hz                                                                                        | 10.0          |
| `check_engage_condition`           | `double` | If false, autonomous transition is always available                                               | 0.1           |
| `nearest_dist_deviation_threshold` | `double` | distance threshold used to find nearest trajectory point                                          | 3.0           |
| `nearest_yaw_deviation_threshold`  | `double` | angle threshold used to find nearest trajectory point                                             | 1.57          |

For `engage_acceptable_limits` related parameters:

| Name                          | Type     | Description                                                                                                                   | Default value |
| :---------------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------- | :------------ |
| `allow_autonomous_in_stopped` | `bool`   | If true, autonomous transition is available when the vehicle is stopped even if other checks fail.                            | true          |
| `dist_threshold`              | `double` | the distance between the trajectory and ego vehicle must be within this distance for `Autonomous` transition.                 | 1.5           |
| `yaw_threshold`               | `double` | the yaw angle between trajectory and ego vehicle must be within this threshold for `Autonomous` transition.                   | 0.524         |
| `speed_upper_threshold`       | `double` | the velocity deviation between control command and ego vehicle must be within this threshold for `Autonomous` transition.     | 10.0          |
| `speed_lower_threshold`       | `double` | the velocity deviation between the control command and ego vehicle must be within this threshold for `Autonomous` transition. | -10.0         |
| `acc_threshold`               | `double` | the control command acceleration must be less than this threshold for `Autonomous` transition.                                | 1.5           |
| `lateral_acc_threshold`       | `double` | the control command lateral acceleration must be less than this threshold for `Autonomous` transition.                        | 1.0           |
| `lateral_acc_diff_threshold`  | `double` | the lateral acceleration deviation between the control command must be less than this threshold for `Autonomous` transition.  | 0.5           |

For `stable_check` related parameters:

| Name                    | Type     | Description                                                                                                                       | Default value |
| :---------------------- | :------- | :-------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| `duration`              | `double` | the stable condition must be satisfied for this duration to complete the transition.                                              | 0.1           |
| `dist_threshold`        | `double` | the distance between the trajectory and ego vehicle must be within this distance to complete `Autonomous` transition.             | 1.5           |
| `yaw_threshold`         | `double` | the yaw angle between trajectory and ego vehicle must be within this threshold to complete `Autonomous` transition.               | 0.262         |
| `speed_upper_threshold` | `double` | the velocity deviation between control command and ego vehicle must be within this threshold to complete `Autonomous` transition. | 2.0           |
| `speed_lower_threshold` | `double` | the velocity deviation between control command and ego vehicle must be within this threshold to complete `Autonomous` transition. | 2.0           |

## Future extensions / Unimplemented parts

- Need to remove backward compatibility interfaces.
- This node should be merged to the `vehicle_cmd_gate` due to its strong connection.
