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
include_directories("nvdla_lib/include")
include_directories("nvdla_lib")

# initial variblles
set(ONNC_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-normal/lib/libonnc.a")
set(ONNX_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx.a")
set(ONNX_PROTO_LINKER_LIBS 
"/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx_proto.a")
set(ONNC_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/onncroot/include")
set(ONNX_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/build-normal/include")
set(ONNX_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/src/include")
set(ONNX_LINKER_INCLUDE 
"/home/dev/Workspace/onnc-umbrella/install-normal/home/dev/Workspace/onnc-umbrella/onnc/include")

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
    nvdla_lib/*.cpp
    nvdla_lib/include/*.hpp
    nvdla_lib/*.c
    )

# Source file lists
file(GLOB RELU_TEST_SRCS
    tools/relu_test.cpp)


file(GLOB CONV_TEST_SRCS
    tools/conv_test.cpp)

# generate executable
add_executable(relu_test ${COMPILER_SRCS} ${RELU_TEST_SRCS})
add_executable(conv_test ${COMPILER_SRCS} ${CONV_TEST_SRCS})

# link libraries
target_link_libraries(relu_test ${ONNC_LINKER_LIBS} 
${ONNX_LINKER_LIBS} 
${ONNX_PROTO_LINKER_LIBS}
${PROTOBUF_LIBRARIES})

target_link_libraries(conv_test ${ONNC_LINKER_LIBS} 
${ONNX_LINKER_LIBS} 
${ONNX_PROTO_LINKER_LIBS}
${PROTOBUF_LIBRARIES})

# Related headers
target_include_directories(relu_test 
    PUBLIC 
    ${ONNC_LINKER_INCLUDE}
    ${ONNX_LINKER_INCLUDE}
)

target_include_directories(conv_test 
    PUBLIC 
    ${ONNC_LINKER_INCLUDE}
    ${ONNX_LINKER_INCLUDE}
)
