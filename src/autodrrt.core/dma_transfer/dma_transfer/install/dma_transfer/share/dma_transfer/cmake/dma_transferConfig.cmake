# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_dma_transfer_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED dma_transfer_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(dma_transfer_FOUND FALSE)
  elseif(NOT dma_transfer_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(dma_transfer_FOUND FALSE)
  endif()
  return()
endif()
set(_dma_transfer_CONFIG_INCLUDED TRUE)

# output package information
if(NOT dma_transfer_FIND_QUIETLY)
  message(STATUS "Found dma_transfer: 0.0.0 (${dma_transfer_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'dma_transfer' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${dma_transfer_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(dma_transfer_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake;ament_cmake_export_include_directories-extras.cmake;ament_cmake_export_libraries-extras.cmake")
foreach(_extra ${_extras})
  include("${dma_transfer_DIR}/${_extra}")
endforeach()
