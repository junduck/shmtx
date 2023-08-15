cmake_minimum_required(VERSION 3.19)
project(shmtx)

add_library(shmtx_lib INTERFACE)
target_include_directories(shmtx_lib INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_executable(shmtx main.cpp)
target_link_libraries(shmtx shmtx_lib)
