cmake_minimum_required(VERSION 3.5)

project(mussp)

unset(CMAKE_CXX_FLAGS CACHE)
unset(CMAKE_CXX_FLAGS_RELEASE CACHE)
unset(CMAKE_CXX_FLAGS_DEBUG CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors -Werror -std=c++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4 -DGTL_STATIC" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  /MD /Ox /Ob2 /Oi /Ot /arch:AVX /fp:fast" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  /MDd /Od /Ob0" CACHE STRING COMPILE_FLAGS FORCE)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

set(MUSSP_SOURCES
  muSSP/Graph.cpp
  muSSP/Node.cpp
  muSSP/Sink.cpp
  muSSP/mussp.cpp
)

set(MUSSP_HEADERS
  muSSP/Graph.h
  muSSP/Node.h
  muSSP/Sink.h
  muSSP/mussp.h
)

ament_auto_add_library(mussp SHARED
  ${MUSSP_SOURCES}
  ${MUSSP_HEADERS}
)

ament_export_include_directories(include)

install(
  FILES ${MUSSP_HEADERS}
  DESTINATION include/mussp
)

ament_auto_package()
