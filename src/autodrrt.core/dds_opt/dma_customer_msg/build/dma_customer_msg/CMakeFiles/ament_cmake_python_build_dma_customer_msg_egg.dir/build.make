# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg

# Utility rule file for ament_cmake_python_build_dma_customer_msg_egg.

# Include any custom commands dependencies for this target.
include CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/progress.make

CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg:
	cd /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/ament_cmake_python/dma_customer_msg && /usr/bin/python3.8 setup.py egg_info

ament_cmake_python_build_dma_customer_msg_egg: CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg
ament_cmake_python_build_dma_customer_msg_egg: CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/build.make
.PHONY : ament_cmake_python_build_dma_customer_msg_egg

# Rule to build all files generated by this target.
CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/build: ament_cmake_python_build_dma_customer_msg_egg
.PHONY : CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/build

CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/clean

CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/depend:
	cd /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg /home/liheyi/swpld/dma_transfer/ros2_pkg/dma_customer_msg/build/dma_customer_msg/CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ament_cmake_python_build_dma_customer_msg_egg.dir/depend

