# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import launch

from .launch_context import LaunchContext


class Machine:
    """Describes a machine for remotely launching ROS nodes."""

    def __init__(self,
                 **kwargs
                 ) -> None:
        """Initialize a machine description."""
        pass

    async def execute_process(self,
                              process_event_args: None,
                              log_cmd: False,
                              emulate_tty: False,
                              shell: False,
                              cleanup_fn: lambda: False,
                              context: LaunchContext) -> None:
        logger = launch.logging.get_logger(process_event_args['name'])
        logger.error("No machine defined")
