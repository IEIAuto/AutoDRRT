add_library(tvm_runtime STATIC IMPORTED)
find_library(tvm_runtime_LIBRARY_PATH tvm_runtime HINTS "${CMAKE_CURRENT_LIST_DIR}/../../")
set_target_properties(tvm_runtime PROPERTIES IMPORTED_LOCATION "${tvm_runtime_LIBRARY_PATH}")
list(APPEND tvm_runtime_LIBRARIES "${tvm_runtime_LIBRARY_PATH}")
