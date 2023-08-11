# tier4_rtc_msgs

## Overview

The `tier4_rtc_msgs` package includes message/service definition files for request to cooperate (RTC).

## Description

### Topic

- `tier4_rtc_msgs/Command`

  - Members
    - `type` : Type of command
  - Constants
    - `DEACTIVATE` : Deactivate command
    - `ACTIVATE` : Activate command

- `tier4_rtc_msgs/CooperateCommand`

  - Members
    - `uuid` : Index of requesting target
    - `module` : Module type of requesting target
    - `command` : Command

- `tier4_rtc_msgs/CooperateResponse`

  - Members
    - `uuid` : Index of requesting target
    - `module` : Module type of requesting target
    - `success` : Cooperate Command result

- `tier4_rtc_msgs/Module`

  - Members
    - `type` : Type of module
  - Constants
    - `NONE`
    - `LANE_CHANGE_LEFT`
    - `LANE_CHANGE_RIGHT`
    - `AVOIDANCE_LEFT`
    - `AVOIDANCE_RIGHT`
    - `PULL_OVER`
    - `PULL_OUT`
    - `TRAFFIC_LIGHT`
    - `INTERSECTION`
    - `CROSSWALK`
    - `BLIND_SPOT`
    - `DETECTION_AREA`
    - `NO_STOPPING_AREA`
    - `OCCLUSION_SPOT`

- `tier4_rtc_msgs/CooperateStatus`

  - Members
    - `stamp` : Time stamp
    - `uuid` : Index of requesting target
    - `module` : Module type of requesting target
    - `safe` : Safety status
    - `command_status` : Received command status
    - `distance` : Distance to requesting target

- `tier4_rtc_msgs/CooperateStatusArray`
  - Members
    - `statuses` : Array of cooperate status message

### Service

- `tier4_rtc_msgs/CooperateCommands`
  - Members (Request)
    - `stamp` : Time stamp
    - `commands` : Array of cooperate command
  - Members (Response)
    - `responses` : Array of cooperate response
