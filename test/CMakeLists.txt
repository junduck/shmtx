cmake_minimum_required(VERSION 3.19)
project(shmtx_test)

add_subdirectory(lib/googletest)

set(TEST_SRC main.cpp)
add_executable(shmtx_test ${TEST_SRC})
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # If it is Clang, add TSAN flags
  target_compile_options(shmtx_test PRIVATE -fsanitize=thread)
  target_link_options(shmtx_test PRIVATE -fsanitize=thread)
endif()
target_link_libraries(shmtx_test gtest gtest_main pthread shmtx_lib)
add_test(shmtx_test shmtx_test)
