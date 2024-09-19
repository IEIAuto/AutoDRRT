# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_io_opt_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED io_opt_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(io_opt_FOUND FALSE)
  elseif(NOT io_opt_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(io_opt_FOUND FALSE)
  endif()
  return()
endif()
set(_io_opt_CONFIG_INCLUDED TRUE)

# output package information
if(NOT io_opt_FIND_QUIETLY)
  message(STATUS "Found io_opt: 0.0.0 (${io_opt_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'io_opt' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${io_opt_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(io_opt_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${io_opt_DIR}/${_extra}")
endforeach()
