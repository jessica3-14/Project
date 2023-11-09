#!/bin/bash

module purge

module load CUDA/9.2.88
module load intel/2020b
module load GCCcore/7.3.0
module load CMake/3.12.1

cmake \
    -Dcaliper_DIR=/home/jessicavwlms/CALI/share/cmake/caliper \
    .

make
