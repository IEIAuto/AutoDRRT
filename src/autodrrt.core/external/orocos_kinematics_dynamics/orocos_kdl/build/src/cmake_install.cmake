# Install script for directory: /data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src

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
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.5.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.5"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "$ORIGIN/../lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/src/liborocos-kdl.so.1.5.1"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/src/liborocos-kdl.so.1.5"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.5.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so.1.5"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "::::::::::::::"
           NEW_RPATH "$ORIGIN/../lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so"
         RPATH "$ORIGIN/../lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/src/liborocos-kdl.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so"
         OLD_RPATH "::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/liborocos-kdl.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kdl" TYPE FILE FILES
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/articulatedbodyinertia.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chain.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chaindynparam.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainexternalwrenchestimator.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainfdsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainfdsolver_recursive_newton_euler.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainfksolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainfksolverpos_recursive.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainfksolvervel_recursive.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainhdsolver_vereshchagin.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainidsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainidsolver_recursive_newton_euler.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainidsolver_vereshchagin.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolverpos_lma.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolverpos_nr.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolverpos_nr_jl.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolvervel_pinv.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolvervel_pinv_givens.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolvervel_pinv_nso.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainiksolvervel_wdls.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainjnttojacdotsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/chainjnttojacsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frameacc.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frameacc.inl"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frameacc_io.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frames.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frames.inl"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/frames_io.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/framevel.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/framevel.inl"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/framevel_io.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/jacobian.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/jntarray.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/jntarrayacc.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/jntarrayvel.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/jntspaceinertiamatrix.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/joint.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/kdl.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/kinfam.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/kinfam_io.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/motion.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_circle.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_composite.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_cyclic_closed.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_line.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_point.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/path_roundedcomposite.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/rigidbodyinertia.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/rotational_interpolation.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/rotational_interpolation_sa.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/rotationalinertia.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/segment.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/solveri.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/stiffness.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/trajectory.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/trajectory_composite.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/trajectory_segment.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/trajectory_stationary.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/tree.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treefksolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treefksolverpos_recursive.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeidsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeidsolver_recursive_newton_euler.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeiksolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeiksolverpos_nr_jl.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeiksolverpos_online.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treeiksolvervel_wdls.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/treejnttojacsolver.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile_dirac.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile_rect.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile_spline.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile_trap.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/velocityprofile_traphalf.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/build/src/config.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kdl/utilities" TYPE FILE FILES
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/error.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/error_stack.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/kdl-config.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/ldl_solver_eigen.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/rall1d.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/rall1d_io.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/rall2d.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/rall2d_io.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/rallNd.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/scoped_ptr.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/svd_HH.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/svd_eigen_HH.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/svd_eigen_Macie.hpp"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/traits.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/utility.h"
    "/data/source/v1.0/autodrrt_v2.0/src/autodrrt.core/external/orocos_kinematics_dynamics/orocos_kdl/src/utilities/utility_io.h"
    )
endif()

