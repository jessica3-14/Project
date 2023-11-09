#!/bin/bash

module purge

module load intel/2020b
module load CMake/3.12.1

cmake \
    -Dcaliper_DIR=~/CALI \
    .

make
