# Planning Validator

The `planning_validator` is a module that checks the validity of a trajectory before it is published. The status of the validation can be viewed in the `/diagnostics` and `/validation_status` topics. When an invalid trajectory is detected, the `planning_validator` will process the trajectory following the selected option: "0. publish the trajectory as it is", "1. stop publishing the trajectory", "2. publish the last validated trajectory".

![planning_validator](./image/planning_validator.drawio.svg)

## Supported features

The following features are supported for trajectory validation and can have thresholds set by parameters:

- **Invalid field** : e.g. Inf, Nan
- **Trajectory points interval** : invalid if any of the distance of trajectory points is too large
- **Curvature** : invalid if the trajectory has too sharp turns that is not feasible for the given vehicle kinematics
- **Relative angle** : invalid if the yaw angle changes too fast in the sequence of trajectory points
- **Lateral acceleration** : invalid if the expected lateral acceleration/deceleration is too large
- **Longitudinal acceleration/deceleration** : invalid if the acceleration/deceleration in the trajectory point is too large
- **Steering angle** : invalid if the expected steering value is too large estimated from trajectory curvature
- **Steering angle rate** : invalid if the expected steering rate value is too large
- **Velocity deviation** : invalid if the planning velocity is too far from the ego velocity
- **Distance deviation** : invalid if the ego is too far from the trajectory

The following features are to be implemented.

- **(TODO) TTC calculation** : invalid if the expected time-to-collision is too short on the trajectory

## Inputs/Outputs

### Inputs

The `planning_validator` takes in the following inputs:

| Name                 | Type                                   | Description                                    |
| -------------------- | -------------------------------------- | ---------------------------------------------- |
| `~/input/kinematics` | nav_msgs/Odometry                      | ego pose and twist                             |
| `~/input/trajectory` | autoware_auto_planning_msgs/Trajectory | target trajectory to be validated in this node |

### Outputs

It outputs the following:

| Name                         | Type                                       | Description                                                               |
| ---------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| `~/output/trajectory`        | autoware_auto_planning_msgs/Trajectory     | validated trajectory                                                      |
| `~/output/validation_status` | planning_validator/PlanningValidatorStatus | validator status to inform the reason why the trajectory is valid/invalid |
| `/diagnostics`               | diagnostic_msgs/DiagnosticStatus           | diagnostics to report errors                                              |

## Parameters

The following parameters can be set for the `planning_validator`:

### System parameters

| Name                               | Type | Description                                                                                                                                                                                                                                | Default value |
| :--------------------------------- | :--- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------ |
| `invalid_trajectory_handling_type` | int  | set the operation when the invalid trajectory is detected. <br>0: publish the trajectory even if it is invalid, <br>1: stop publishing the trajectory, <br>2: publish the last validated trajectory.                                       | 0             |
| `publish_diag`                     | bool | the Diag will be set to ERROR when the number of consecutive invalid trajectory exceeds this threshold. (For example, threshold = 1 means, even if the trajectory is invalid, the Diag will not be ERROR if the next trajectory is valid.) | true          |
| `diag_error_count_threshold`       | int  | if true, diagnostics msg is published.                                                                                                                                                                                                     | true          |
| `display_on_terminal`              | bool | show error msg on terminal                                                                                                                                                                                                                 | true          |

### Algorithm parameters

#### Thresholds

The input trajectory is detected as invalid if the index exceeds the following thresholds.

| Name                              | Type   | Description                                                                                                   | Default value |
| :-------------------------------- | :----- | :------------------------------------------------------------------------------------------------------------ | :------------ |
| `thresholds.interval`             | double | invalid threshold of the distance of two neighboring trajectory points                                        | 100.0         |
| `thresholds.relative_angle`       | double | invalid threshold of the relative angle of two neighboring trajectory points                                  | 2.0           |
| `thresholds.curvature`            | double | invalid threshold of the curvature in each trajectory point                                                   | 1.0           |
| `thresholds.lateral_acc`          | double | invalid threshold of the lateral acceleration in each trajectory point                                        | 9.8           |
| `thresholds.longitudinal_max_acc` | double | invalid threshold of the maximum longitudinal acceleration in each trajectory point                           | 9.8           |
| `thresholds.longitudinal_min_acc` | double | invalid threshold of the minimum longitudinal deceleration in each trajectory point                           | -9.8          |
| `thresholds.steering`             | double | invalid threshold of the steering angle in each trajectory point                                              | 1.414         |
| `thresholds.steering_rate`        | double | invalid threshold of the steering angle rate in each trajectory point                                         | 10.0          |
| `thresholds.velocity_deviation`   | double | invalid threshold of the velocity deviation between the ego velocity and the trajectory point closest to ego. | 100.0         |
| `thresholds.distance_deviation`   | double | invalid threshold of the distance deviation between the ego position and the trajectory point closest to ego. | 100.0         |
