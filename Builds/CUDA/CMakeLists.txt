cmake_minimum_required(VERSION 3.12)

project(bucket LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)

add_executable(bucket.out bucket.cu)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})

target_link_libraries(bucket.out PRIVATE MPI::MPI_CXX)
target_link_libraries(bucket.out PRIVATE caliper)
