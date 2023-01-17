cmake_minimum_required(VERSION 3.19)
project(shmtx_test)

add_subdirectory(lib/googletest)

set(TEST_SRC main.cpp)
add_executable(shmtx_test ${TEST_SRC})
target_link_libraries(shmtx_test gtest gtest_main pthread shmtx_lib)
add_test(shmtx_test shmtx_test)
