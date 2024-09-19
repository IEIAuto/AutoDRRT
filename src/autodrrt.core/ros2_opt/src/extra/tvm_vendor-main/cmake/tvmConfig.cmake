add_library(tvm STATIC IMPORTED)
find_library(tvm_LIBRARY_PATH tvm HINTS "${CMAKE_CURRENT_LIST_DIR}/../../")
set_target_properties(tvm PROPERTIES IMPORTED_LOCATION "${tvm_LIBRARY_PATH}")
list(APPEND tvm_LIBRARIES "${tvm_LIBRARY_PATH}")
