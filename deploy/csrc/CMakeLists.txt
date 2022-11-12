if (NOT MSVC)
    set(CMAKE_CXX_STANDARD 14)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# build ONNXRUNTIME ops
if (NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif ()
if (NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR " ONNXRUNTIME_DIR is not found.")
else ()
    message(STATUS "Build ONNXRUNTIME custom ops.")
    add_subdirectory(onnxruntime)
endif ()

# build TensorRT ops
if (NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
endif ()
message(STATUS "Build TensorRT custom ops.")
add_subdirectory(tensorrt)
