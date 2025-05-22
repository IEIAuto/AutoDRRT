ssh_machine
===========

This package provides and implementation of the `launch.Machine` class that
can be used to launch processes on remote machines via SSH.  It is intended
to be used together with the `multi-machine-launching` branch in my fork of
the `launch` repository:
https://github.com/pjreed/launch/tree/multi-machine-launching

Prerequisites
-------------

This has been tested in ROS 2 Foxy on Ubuntu 20.04.  It
also requires the `python3-asyncssh` package.

Example Usage
-------------

Create a workspace, or clone these repos into an existing one:

```bash
# Set up the workspace
. /opt/ros/foxy/setup.bash
mkdir -p ssh_machine_ws/src
cd ssh_machine_ws/src
git clone -b multi-machine-launching https://github.com/pjreed/launch.git
git clone https://github.com/pjreed/ssh_machine.git

# Install dependencies
rosdep install . -y --from-paths -i --os=ubuntu:bionic --skip-keys python3-asyncssh

# Build it
cd ..
colcon build --merge-install
. install/setup.bash

# Run the demo; edit ssh_machine/launch/pub_sub.launch.py to use a machine you can access
ros2 launch ssh_machine pub_sub.launch.py
```

How It Works
------------

In my fork of the `launch` repository, the `launch.actions.ExecuteProcess` class
has been modified to accept a `machine=` argument that defines how the process
should be launched.  That repo also contains a `launch.LocalMachine` class that
implements the previous functionality and will launch a process on the local host;
this is the default behavior if no machine is specified.

This repository defines an `ssh_machine.SshMachine` class that uses the `asyncssh`
library to launch processes on a remote host.  It works by creating an SSH channel
to the configured host, then running processes in the background; when it is
told to shut down, it kills all background processes and then exits.  It takes
two arguments; a `hostname` that specifies the host name (or IP) of the remote
host, and an `env` that specifies a script that should be sourced to configure
the environment (for example, `/opt/ros/dashing/setup.bash`).

Limitations
-----------

This is a naive implementation that needs to be made more robust for practical use.
Currently:

- The paths to ROS nodes are resolved before the command is sent over the SSH
  channel, meaning nodes must be installed on both the local and remote host and
  they must be in the same path.
    - A side effect of this is that you must have the same ROS distrubtion
      installed on every host; for example, if you try to launch a node with the
      package `demo_nodes_cpp` and executable `talker` on a remote host, if the
      local host is using ROS Foxy, then it will be resolved to
      `/opt/ros/foxy/lib/demo_nodes_cpp/talker` before it is executed on the
      remote host.
- The ssh connection is not very customizable; it will connect as the user who
  is running the launch script, and it will not prompt the user for a password,
  so you must have public key authentication correctly configured.
- Aside from specifying a script to source, the environment can't be configured.
  It should be possible to configure the environment so that it is possible to,
  for example, change the DDS implementation or domain ID.
