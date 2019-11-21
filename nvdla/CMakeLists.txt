cmake_minimum_required(VERSION 3.2)
project(nvdla_lib)

# Glog functions
include(cmake/FindGlog.cmake)

####################
#  External
#  Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

#  glog
#find_package(Glog REQUIRED)

include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIBRARY_DIR})

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")

# initial variblles
set(ONNC_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-dbg/lib/libonnc.a")
set(ONNX_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx.a")
set(ONNX_PROTO_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx_proto.a")
set(ONNC_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/onncroot/include")
set(ONNX_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/install-dbg/home/dev/Workspace/onnc-umbrella/build/include")


# Generic compilation options
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message("Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS} -rdynamic")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS} -rdynamic")
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
    src/*.cpp
    src/*.c
    src/nvdla_op/*.cpp)


# Source file lists
file(GLOB RELU_TEST_SRCS
    tools/relu_test.cpp)

file(GLOB READ_PROTO_SRCS
    tools/read_proto.cpp)


# generate executable
add_executable(relu_test ${COMPILER_SRCS} ${RELU_TEST_SRCS})
add_executable(read_proto ${COMPILER_SRCS} ${READ_PROTO_SRCS})

# link libraries
target_link_libraries(relu_test ${ONNC_LINKER_LIBS} 
${ONNX_LINKER_LIBS} 
${ONNX_PROTO_LINKER_LIBS}
${PROTOBUF_LIBRARIES})

target_link_libraries(read_proto ${ONNC_LINKER_LIBS} 
${ONNX_LINKER_LIBS} 
${ONNX_PROTO_LINKER_LIBS}
${PROTOBUF_LIBRARIES})

# Related headers
target_include_directories(relu_test 
    PUBLIC 
    ${ONNC_LINKER_INCLUDE}
    ${ONNX_LINKER_INCLUDE}
)
target_include_directories(read_proto 
    PUBLIC 
    ${ONNC_LINKER_INCLUDE}
    ${ONNX_LINKER_INCLUDE}
)
