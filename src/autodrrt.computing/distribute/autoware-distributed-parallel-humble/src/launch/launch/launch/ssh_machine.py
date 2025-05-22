# Copyright 2019 Southwest Research Institute
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
from asyncio import Event
from logging import Logger

import asyncssh
import traceback
from typing import Optional, cast

# from launch import SomeActionsType, EventHandler
# from launch.actions import OpaqueFunction
# from launch.event_handlers import OnShutdown
# from launch.events.process import ProcessStarted, ProcessExited, ShutdownProcess, \
#     SignalProcess
# from launch.launch_context import LaunchContext
# from .machine import Machine
# from launch.some_substitutions_type import SomeSubstitutionsType
# from launch.substitution import Substitution
# from launch.utilities import is_a_subclass

from .event_handler import EventHandler
from .some_actions_type import SomeActionsType
from .actions.opaque_function import OpaqueFunction
from .events.process import ProcessExited
from .events.process import ProcessIO
from .events.process import ProcessStarted
from .events.process import ProcessStderr
from .events.process import ProcessStdin
from .events.process import ProcessStdout
from .events.process import ShutdownProcess
from .events.process import SignalProcess
from .launch_context import LaunchContext
from .event_handlers import OnShutdown
from .machine import Machine
from .some_substitutions_type import SomeSubstitutionsType
from .substitution import Substitution
from .utilities import is_a_subclass

import launch


class SshClientSession(asyncssh.SSHClientSession):
    """
    Factory for generating SSH client sessions
    """
    def __init__(self, logger: Logger, context: LaunchContext, process_event_args=None):
        self.__logger = logger
        self.__context = context
        self.__process_event_args = process_event_args

    def connection_made(self, chan):
        self.__logger.debug("connection_made")

    def data_received(self, data, datatype):
        # Probably should emit this data via an event for the launch system
        self.__logger.info("data_received: %s" % str(data))

    def connection_lost(self, exc):
        self.__logger.debug("connection_lost: %s" % exc)


class SshMachine(Machine):
    """Describes a machine for remotely launching ROS nodes."""

    def __init__(self, *,
                 hostname: SomeSubstitutionsType,
                 env: Optional[SomeSubstitutionsType] = None,
                 info:None,
                 **kwargs) -> None:
        """Initialize a machine description."""
        super().__init__(**kwargs)
        self.__hostname = hostname
        self.__env = env
        self.__conn = None
        self.__chan = None
        self.__logger = None
        self.__first_run = True
        self.__connection_ready = asyncio.Event()
        self.__info = info

    @property
    def hostname(self) -> Substitution:
        return self.__hostname

    @property
    def env(self):
        return self.__env

    def __on_signal_process_event(self, event: Event, context: LaunchContext):
        if self.__chan:
            typed_event = cast(SignalProcess, context.locals.event)
            self.__logger.info("signals don't work on OpenSSH < 7.9")
            self.__chan.signal(typed_event.signal_name)

    def __on_shutdown(self, event: Event, context: LaunchContext) -> Optional[SomeActionsType]:
        try:
            if self.__chan:
                self.__logger.debug("Killing all jobs")
                self.__chan.write('kill $(jobs -p)')
                self.__chan.write_eof()
                self.__chan.close()
            self.__logger.debug("Closing SSH connection")
            self.__conn.close()
        except Exception:
            self.__logger.error("Exception when shutting down channel: %s" % traceback.format_exc())

    async def execute_process(self,
                              process_event_args: None,
                              log_cmd: False,
                              emulate_tty: False,
                              shell: False,
                              cleanup_fn: lambda: False,
                              context: LaunchContext) -> None:
        if process_event_args is None:
            raise RuntimeError('process_event_args unexpectedly None')
        cmd = process_event_args['cmd']
        cwd = process_event_args['cwd']
        env = process_event_args['env']
    
        
        if not self.__logger:
            # The first time this method is called, set up a logger and
            # event handlers for it.
            self.__logger = launch.logging.get_logger(process_event_args['name'])

            event_handlers = [
                EventHandler(
                    matcher=lambda event: is_a_subclass(event, ShutdownProcess),
                    entities=OpaqueFunction(function=self.__on_shutdown),
                ),
                EventHandler(
                    matcher=lambda event: is_a_subclass(event, SignalProcess),
                    entities=OpaqueFunction(function=self.__on_signal_process_event),
                ),
                OnShutdown(on_shutdown=self.__on_shutdown)
            ]

            self.__logger.debug("Registering event handlers")
            for handler in event_handlers:
                context.register_event_handler(handler)

        if log_cmd:
            self.__logger.info("process details: cmd=[{}], cwd='{}', custom_env?={}".format(
                ', '.join(cmd), cwd, 'True' if env is not None else 'False'
            ))

        self.__logger.debug("Executing process")

        process_event_args['pid'] = 0
        await context.emit_event(ProcessStarted(**process_event_args))

        try:
            if self.__first_run:
                # The first time this method runs, create an SSH connection
                # and initialize the environment.
                self.__first_run = False

                def create_session():
                    return SshClientSession(self.__logger, context, process_event_args)
                self.__conn = await asyncio.wait_for(asyncssh.connect(self.__info["ip"], username=self.__info["username"], port=19010, known_hosts=None),timeout=5000)
                # await asyncio.sleep(5)
                self.__chan, session = await self.__conn.create_session(
                    create_session,
                    encoding='utf8')
                
                if self.__env:
                    self.__chan.write(self.__env)
                    self.__chan.write('\n')
                self.__connection_ready.set()

            # Every other time this method is called, we need to wait until
            # the environment is ready.
            await self.__connection_ready.wait()
            if self.__chan:
                # Run the command and put it in the background, then wait until
                # the SSH channel closes
                
                # if cmd[0]=="/opt/ros/humble/lib/rclcpp_components/component_container":
                #     cmd[0] = "/opt/ros/humble/install/rclcpp_components/lib/rclcpp_components/component_container"
                # elif cmd[0]=="/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher":
                #     cmd[0] = "/opt/ros/humble/install/robot_state_publisher/lib/robot_state_publisher/robot_state_publisher"
                # elif cmd[0]=="/opt/ros/humble/lib/rclcpp_components/component_container_mt":
                #     cmd[0] = "/opt/ros/humble/install/rclcpp_components/lib/rclcpp_components/component_container_mt"
                # elif cmd[0]=="/opt/ros/humble/lib/diagnostic_aggregator/aggregator_node":
                #     cmd[0] = "/opt/ros/humble/install/diagnostic_aggregator/lib/aggregator_node/aggregator_node"
                # elif cmd[0]=="/opt/ros/humble/lib/demo_nodes_cpp/talker":
                #     cmd[0] = "/opt/ros/humble/install/demo_nodes_cpp/lib/demo_nodes_cpp/talker"
                cmd_tmp= []
                dict_tmp = {"node_name":"", "name_space":""}
                for x in cmd:
                    if x.startswith('__node:='):
                        dict_tmp["node_name"] = x[8:]
                    if x.startswith("__ns:="):
                        dict_tmp["name_space"] = x[6:]
                    if x.startswith("~/"):
                        x = dict_tmp["name_space"] + '/' + dict_tmp["node_name"]  + x[1:]
                    cmd_tmp.append(x)
                
                self.__chan.write(' '.join(cmd_tmp) + ' &\n')
                
            
                # print("==========================================================================")
                # print(self.__info["ip"])
                # print(' '.join(cmd_tmp) + ' &\n')
                # with open("/home/orin/autoware/log.txt","a") as f:
                #     f.write(' '.join(cmd) + ' &\n')
                # print("==========================================================================")
                
                            
                self.__logger.debug("Waiting for SSH channel to close")
                await self.__chan.wait_closed()

                await context.emit_event(ProcessExited(
                    returncode=self.__chan.get_exit_status(),
                    **process_event_args))

                self.__logger.debug("SSH connection exiting")
            else:
                self.__logger.error("SSH channel wasn't ready")
        except Exception:
            self.__logger.error('exception occurred while executing process:\n{}'.format(
                traceback.format_exc()
            ))
        finally:
            cleanup_fn()
