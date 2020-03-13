cmake_minimum_required(VERSION 3.2)
project(nvdla_lib)

# Glog functions
include(cmake/FindGlog.cmake)

####################
#  External
#  Protobuf
#find_package(Protobuf REQUIRED)
#include_directories(${PROTOBUF_INCLUDE_DIR})
#  glog
#find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIBRARY_DIR})

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("nvdla_lib/include")
include_directories("include")
include_directories("/home/dev/Workspace/tvm/include")
include_directories("/home/dev/Workspace/tvm/3rdparty/dlpack/include")

# initial variblles
set(ONNC_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/install-dbg/lib/libonnc.so")
set(ONNX_LINKER_LIBS 
#"/home/dev/Workspace/onnc-umbrella/onncroot/lib/libonnx.a")
"")
set(ONNX_PROTO_LINKER_LIBS 
"")
#"/home/dev/Workspace/onnc-umbrella/onncroot/lib/libonnx_proto.a")
set(ONNC_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/onncroot/include"
"/home/dev/Workspace/onnc-umbrella/install-dbg/include")

# Generic compilation options
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message("Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS} -rdynamic -DNVDLA_DEBUG")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS} -rdynamic -DNVDLA_DEBUG")
else()
    set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
    if (HIDE_PRIVATE_SYMBOLS)
      message(STATUS "Hide private symbols...")
      set(CMAKE_C_FLAGS "-fvisibility=hidden ${CMAKE_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "-fvisibility=hidden ${CMAKE_CXX_FLAGS}")
    endif(HIDE_PRIVATE_SYMBOLS)
  endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++14"    SUPPORT_CXX14)
message(STATUS "Build with c++14")
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")


# add source group
file(GLOB COMPILER_SRCS
    nvdla_lib/*.cpp
    nvdla_lib/include/*.hpp
    src/*.cpp
    src/opt_pass/*.cpp
    )
list(REMOVE_ITEM COMPILER_SRCS "src/NvDlaRuntime.cpp")

# Source file lists
file(GLOB RUNTIME_SRCS
    src/NvDlaRuntime.cpp)

file(GLOB RELU_TEST_SRCS_C
    tests/relu/relu_test.c)

file(GLOB ADD_TEST_SRCS_C
    tests/add/add_test.c)

file(GLOB GLOBAL_AVERAGEPOOL_TEST_SRCS_C
    tests/global_average_pool_test/global_average_pool.c)

file(GLOB AVERAGEPOOL_TEST_SRCS_C
    tests/averagepool/averagepool_test.c)

file(GLOB GEMM_TEST_SRCS_C 
    tests/gemm/gemm_test.c)

file(GLOB MAX_POOL_TEST_SRCS_C 
    tests/max_pool/max_pool_test.c)

file(GLOB BATCHNORM_TEST_SRCS_C
    tests/batchnorm/batchnorm_test.c)

file(GLOB RESHAPE_TEST_SRCS_C 
    tests/reshape/reshape_test.c)

file(GLOB TEST_ADD_FLOAT_WEIGHT_TENSOR
    tests/unittest/test_add_float_weight_tensor.c)   

# generate nvdla library runtime 
add_library(nvdla_library SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
#set_property(TARGET nvdla_library PROPERTY POSITION_INDEPENDENT_CODE ON)

# generate executable
#add_executable(relu_test_c ${RELU_TEST_SRCS_C})
#add_executable(add_test_c ${ADD_TEST_SRCS_C})
#add_executable(test_add_float_weight_tensor ${TEST_ADD_FLOAT_WEIGHT_TENSOR})
#add_executable(averagepool_test_c ${AVERAGEPOOL_TEST_SRCS_C})
#add_executable(max_pool_c ${MAX_POOL_TEST_SRCS_C})
#add_executable(gemm_test_c ${GEMM_TEST_SRCS_C})
#add_executable(reshape_test_c ${RESHAPE_TEST_SRCS_C})
#add_executable(batchnorm_test_c ${BATCHNORM_TEST_SRCS_C})
#add_executable(global_average_pool_test_c ${GLOBAL_AVERAGEPOOL_TEST_SRCS_C})

# link libraries
target_link_libraries(nvdla_library
${ONNC_LINKER_LIBS} 
${ONNX_LINKER_LIBS} 
${ONNX_PROTO_LINKER_LIBS}
${PROTOBUF_LIBRARIES})

#target_link_libraries(relu_test_c nvdla_library)
#target_link_libraries(add_test_c nvdla_library)
#target_link_libraries(averagepool_test_c nvdla_library)
#target_link_libraries(gemm_test_c nvdla_library)
#target_link_libraries(max_pool_c nvdla_library)
#target_link_libraries(reshape_test_c nvdla_library)
#target_link_libraries(batchnorm_test_c nvdla_library)
#target_link_libraries(test_add_float_weight_tensor nvdla_library)
#target_link_libraries(global_average_pool_test_c nvdla_library)

# Related headers
target_include_directories(nvdla_library 
    PUBLIC
    ${ONNC_LINKER_INCLUDE}
)