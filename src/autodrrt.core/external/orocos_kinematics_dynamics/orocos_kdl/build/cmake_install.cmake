# Install script for directory: /data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/cmake/FindEigen3.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/orocos_kdl-config.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/orocos_kdl-config-version.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake/OrocosKDLTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake/OrocosKDLTargets.cmake"
         "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/CMakeFiles/Export/e9ee293449b431876305ebf8d5a6eef6/OrocosKDLTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake/OrocosKDLTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake/OrocosKDLTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/CMakeFiles/Export/e9ee293449b431876305ebf8d5a6eef6/OrocosKDLTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/orocos_kdl/cmake" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/CMakeFiles/Export/e9ee293449b431876305ebf8d5a6eef6/OrocosKDLTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/orocos-kdl.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/orocos_kdl.pc")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/doc/cmake_install.cmake")
  include("/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/src/cmake_install.cmake")
  include("/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/tests/cmake_install.cmake")
  include("/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/models/cmake_install.cmake")
  include("/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
