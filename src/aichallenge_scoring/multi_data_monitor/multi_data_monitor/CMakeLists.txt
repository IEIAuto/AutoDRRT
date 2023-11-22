cmake_minimum_required(VERSION 3.14)
project(multi_data_monitor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Qt5 ${rviz_QT_VERSION} REQUIRED Core Widgets)
set(QT_LIBRARIES Qt5::Widgets)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_compile_definitions(FMT_HEADER_ONLY)

ament_auto_add_library(${PROJECT_NAME}_core SHARED
  "src/core/common/path.cpp"
  "src/core/common/yaml.cpp"
  "src/core/config/types.cpp"
  "src/core/config/plantuml.cpp"
  "src/core/header/filter.cpp"
  "src/core/header/widget.cpp"
  "src/core/parser/check_system_class.cpp"
  "src/core/parser/file.cpp"
  "src/core/parser/construction.cpp"
  "src/core/parser/subscription.cpp"
  "src/core/parser/resolve_relation.cpp"
  "src/core/parser/stylesheet.cpp"
  "src/core/loader/config_loader.cpp"
  "src/core/loader/filter_loader.cpp"
  "src/core/loader/stream_loader.cpp"
  "src/core/loader/widget_loader.cpp"
  "src/core/runner/rclcpp_runner.cpp"
  "src/core/runner/stream_runner.cpp"
  "src/core/runner/widget_runner.cpp"
  "src/core/stream/apply.cpp"
  "src/core/stream/basic.cpp"
  "src/core/stream/field.cpp"
  "src/core/stream/panel.cpp"
  "src/core/stream/print.cpp"
  "src/core/stream/topic.cpp"
  "src/core/filter/function.cpp"
)
target_include_directories(${PROJECT_NAME}_core PRIVATE src/core)

ament_auto_add_library(${PROJECT_NAME}_rviz SHARED
  "src/core/rviz/rviz_manager.hpp"
  "src/core/rviz/rviz_manager.cpp"
  "src/core/rviz/rviz_panel.hpp"
  "src/core/rviz/rviz_panel.cpp"
)
target_include_directories(${PROJECT_NAME}_rviz PRIVATE src/core)

ament_auto_add_library(${PROJECT_NAME}_plugin SHARED
  "src/plugin/widget/matrix.cpp"
  "src/plugin/widget/simple.cpp"
  "src/plugin/widget/titled.cpp"
  "src/plugin/filter/components/conditions.cpp"
  "src/plugin/filter/access.cpp"
  "src/plugin/filter/diag.cpp"
  "src/plugin/filter/dump.cpp"
  "src/plugin/filter/units.cpp"
  "src/plugin/filter/precision.cpp"
  "src/plugin/filter/set_if.cpp"
)
target_link_libraries(${PROJECT_NAME}_plugin ${PROJECT_NAME}_core)

ament_auto_add_executable(parser src/tool/parser.cpp)
target_include_directories(parser PRIVATE src/core)

ament_auto_add_executable(runner src/tool/runner.cpp)
target_include_directories(runner PRIVATE src/core)

pluginlib_export_plugin_description_file(rviz_common plugins/rviz.xml)
pluginlib_export_plugin_description_file(multi_data_monitor plugins/builtins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_gtest test/test.cpp)
  target_compile_definitions(${PROJECT_NAME}_gtest PRIVATE TEST_YAML="${CMAKE_CURRENT_SOURCE_DIR}/test/yaml/")
  target_include_directories(${PROJECT_NAME}_gtest PRIVATE src/core)
  target_link_libraries(${PROJECT_NAME}_gtest ${PROJECT_NAME}_core)
endif()

install(PROGRAMS
  scripts/example
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE css documents icons plugins config)
