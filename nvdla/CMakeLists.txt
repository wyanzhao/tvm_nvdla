cmake_minimum_required(VERSION 3.5)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

####################
#  General settings
#include(option)
#option_enum(NAME CMAKE_BUILD_TYPE HELP "Choose the type of build" VALUE Normal Debug Release)
option(ENABLE_PTHREAD "use pthreads" ON)
set(HAVE_PTHREADS ${ENABLE_PTHREAD})

####################
#  Language
set(STDC_HEADERS 1)
set(CMAKE_CXX_STANDARD 14)
set(HAVE_CXX14 TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g3"          CACHE STRING "Flags used by the CXX compiler during DEBUG builds.")
set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG"     CACHE STRING "Flags used by the CXX compiler during RELEASE builds.")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DNDEBUG -g3" CACHE STRING "Flags used by the CXX compiler during RELWITHDEBINFO builds.")
option(ENABLE_WERROR "Compile with -Werror enabled (default is OFF)" OFF)
if (ENABLE_WERROR)
    add_compile_options(-Werror)
endif()
####################
#  Project
project(nvdla_test)
project(read_proto)

####################
#  External
#  Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

#  glog
find_package(Glog REQUIRED)
message(STATUS "[${PACKAGE}] Using glog include at ${GLOG_INCLUDE_DIR}")

include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIBRARY_DIR})

# nvdla_Test config
add_executable(nvdla_test nvdla_test.cpp src/fp16.c src/nvdla_lib.cpp src/nvdla_meta.cpp src/loadable.cpp)

target_link_libraries(nvdla_test
    "/home/dev/Workspace/onnc-umbrella/build-dbg/lib/libonnc.a")
target_link_libraries(nvdla_test
    "/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx.a")
target_link_libraries(nvdla_test
    "/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx_proto.a")
target_link_libraries(nvdla_test ${PROTOBUF_LIBRARIES})


target_include_directories(nvdla_test PUBLIC "include")
target_include_directories(nvdla_test PUBLIC "/home/dev/Workspace/onnc-umbrella/onncroot/include")
target_include_directories(nvdla_test PUBLIC "/home/dev/Workspace/onnc-umbrella/install-dbg/home/dev/Workspace/onnc-umbrella/build/include")


# read_proto config

add_executable(read_proto read_proto.cpp)

target_link_libraries(read_proto
    "/home/dev/Workspace/onnc-umbrella/build-dbg/lib/libonnc.a")
target_link_libraries(read_proto
    "/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx.a")
target_link_libraries(read_proto
    "/home/dev/Workspace/onnc-umbrella/build-onnx/libonnx_proto.a")
target_link_libraries(read_proto ${PROTOBUF_LIBRARIES})


target_include_directories(read_proto PUBLIC "include")
target_include_directories(read_proto PUBLIC "/home/dev/Workspace/onnc-umbrella/onncroot/include")
target_include_directories(read_proto PUBLIC "/home/dev/Workspace/onnc-umbrella/install-dbg/home/dev/Workspace/onnc-umbrella/build/include")