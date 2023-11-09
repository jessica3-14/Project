cmake_minimum_required(VERSION 3.12)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)

add_executable(bucket.out bucket.cpp)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")

include_directories(${caliper_INCLUDE_DIR})

target_link_libraries(bucket.out PRIVATE MPI::MPI_CXX)
target_link_libraries(bucket.out PRIVATE caliper)

