# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_gpudirect-dds_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED gpudirect-dds_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(gpudirect-dds_FOUND FALSE)
  elseif(NOT gpudirect-dds_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(gpudirect-dds_FOUND FALSE)
  endif()
  return()
endif()
set(_gpudirect-dds_CONFIG_INCLUDED TRUE)

# output package information
if(NOT gpudirect-dds_FIND_QUIETLY)
  message(STATUS "Found gpudirect-dds: 0.20.5 (${gpudirect-dds_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'gpudirect-dds' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${gpudirect-dds_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(gpudirect-dds_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${gpudirect-dds_DIR}/${_extra}")
endforeach()
