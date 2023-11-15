#include <mpi.h>
#include <stdio.h>
#include <random>
#include <cfloat>

#include <iostream>

#include <cstdlib>

#include <ctime>

#include <cuda_runtime.h>
#include <curand_kernel.h>


// CUDA kernel to generate random numbers

__global__ void generateRandomNumbers(double *data, int n,curandState s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
//curandState state;
    curand_init(0, idx, 0, &s);
        data[idx] = static_cast<double>(curand_uniform(&s)) * 100000;
    }
}

__global__ void generateSortedNumbers(double *data, int n,curandState state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
    curand_init(0, idx, 0, &state);
        data[idx] = 100000.0/(n+1) *idx + static_cast<double>(curand_uniform(&state)) * 100000.0/(n+1);
    }
}

__global__ void generateReverseNumbers(double *data, int n,curandState state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
    curand_init(0, idx, 0, &state);
        data[idx] = 100000.0/(n+1) *(n-idx) + static_cast<double>(curand_uniform(&state))  * 100000.0/(n+1);
    }
}
__global__ void perturb1(double *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx%100==0) {
	double temp = data[idx];
        data[idx] = data[idx+1];
	data[idx+1] = temp;
    }
}

void genData(int dataSize, int mode, double* final_arr) {


    double* deviceData;

    cudaMalloc((void**)&deviceData, dataSize * sizeof(double));

    // Set up the random number generator on the host

    std::srand(static_cast<unsigned>(std::time(nullptr)));



    // Set up the CUDA grid and block sizes

    int blockSize = 256;

    int gridSize = (dataSize + blockSize - 1) / blockSize;
curandState state;


    // Launch the CUDA kernel to generate random numbers in parallel

if(mode==0){
    generateRandomNumbers<<<gridSize, blockSize>>>(deviceData, dataSize,state);
}
else if(mode==1){
    generateSortedNumbers<<<gridSize, blockSize>>>(deviceData, dataSize,state);
if(mode==3){
    perturb1<<<gridSize,blockSize>>>(deviceData,dataSize);
}
}
else if(mode==2){
    generateReverseNumbers<<<gridSize, blockSize>>>(deviceData, dataSize,state);
}


    // Copy the generated random data back to the host

    cudaMemcpy(final_arr, deviceData, dataSize * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(deviceData);



}

