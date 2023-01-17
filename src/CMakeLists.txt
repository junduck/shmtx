cmake_minimum_required(VERSION 3.19)
project(shmtx)

# for boost/predef.h maybe we can implement it ourselves to avoid this dependency
find_package(Boost REQUIRED)

add_library(shmtx_lib INTERFACE)
target_include_directories(shmtx_lib INTERFACE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

add_executable(shmtx main.cpp)
target_link_libraries(shmtx shmtx_lib)
