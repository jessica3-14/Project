/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;
  float kcalls;
  size_t size;
void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void quick_sort_step(float *dev_values, int l, int h)
{
  //CALI_MARK_BEGIN(bitonic_sort_step_region);
  
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (i >= l && i <=h){
    float piv = dev_values[l];
    int left = l;
    int right = h;
    
    while (left <= right){
        while (left <= h && dev_values[left] <= piv){
            left++;
        }
        while (right >= l && dev_values[right] > piv){
            right--;
        }
        
        if (left <= right) {
                // Swap elements if necessary
                float temp = dev_values[left];
                dev_values[left] = dev_values[right];
                dev_values[right] = temp;
                left++;
                right--;
        }
    }
  }
  
  //CALI_MARK_END(bitonic_sort_step_region);
}

/**
 * Inplace bitonic sort using CUDA.
 */
void quick_sort(float *values, int low, int high)
{
  float *dev_values;
  size_t size = (high - low + 1) * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEvent_t startcudamcpyhtd,stopcudamcpyhtd;
  float ms = 0.0;
  cudaEventCreate(&startcudamcpyhtd);
  cudaEventCreate(&stopcudamcpyhtd);
  cudaEventRecord(startcudamcpyhtd);
  
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  
  cudaEventRecord(stopcudamcpyhtd);
  cudaEventSynchronize(stopcudamcpyhtd);
  cudaEventElapsedTime(&ms,startcudamcpyhtd,stopcudamcpyhtd);
  CALI_MARK_END(cudaMemcpy_host_to_device);
  cudaMemcpy_host_to_device_time = ms/1000.0;

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEvent_t startk,stopk;
  cudaEventCreate(&startk);
  cudaEventCreate(&stopk);
  cudaEventRecord(startk);
  
  /* Major step */
  quick_sort_step<<<blocks, threads>>>(dev_values, 0, size - 1);
  
  cudaDeviceSynchronize();
  cudaEventRecord(stopk);
  cudaEventSynchronize(stopk);
  cudaEventElapsedTime(&ms,startk,stopk);
  bitonic_sort_step_time = ms/1000.0;
  CALI_MARK_END(bitonic_sort_step_region);
 
  
  //MEM COPY FROM DEVICE TO HOST
 CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEvent_t startcudamcpydth,stopcudamcpydth;
  cudaEventCreate(&startcudamcpydth);
  cudaEventCreate(&stopcudamcpydth);
  cudaEventRecord(startcudamcpydth);
  
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(stopcudamcpydth);
  cudaEventSynchronize(stopcudamcpydth);
  cudaEventElapsedTime(&ms,startcudamcpydth,stopcudamcpydth);
  cudaMemcpy_device_to_host_time = ms / 1000.0;
 CALI_MARK_END(cudaMemcpy_device_to_host);
  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
  
  cali::ConfigManager mgr;
  mgr.start();
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  
  clock_t start, stop;
  start = clock();
  quick_sort(values, 0, NUM_VALS-1);
  stop = clock();
  
  print_elapsed(start,stop);
  float data_size = kcalls * size * 4 * (1e-9);
  //printf("Effective Bandwidth: %.6fGB/s\n",(data_size/bitonic_sort_step_time));
  printf("Bitonic sort step time: %fs\n",bitonic_sort_step_time);
  printf("cudaMemcpy_host_to_device_time: %fs\n", cudaMemcpy_host_to_device_time);
  printf("cudaMemcpy_device_to_host_time: %fs\n", cudaMemcpy_device_to_host_time);
  mgr.stop();
  mgr.flush();
 
}