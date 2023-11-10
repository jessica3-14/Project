#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include<caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

__device__ int d_NUM_VALS; // Declare as a device constant

//const char* bubble_sort_step_region = "bubble_sort_step";
//const char* entire_computation_region = "entire_computation";
int kernel_call;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

int verify(float *values){
  for(int i = 0; i < sizeof(values)-1; i++){
    if(values[i] > values[i+1]){
      return -1;
    }
  }
  return 1;
}

__global__ void bubble_sort_step(float *dev_values)
{
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int next = i + 1;

  if (next < d_NUM_VALS) {
    if (dev_values[i] > dev_values[next]) {
      float temp = dev_values[i];
      dev_values[i] = dev_values[next];
      dev_values[next] = temp;
    }
  }
}

void bubble_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**)&dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  // Copy NUM_VALS to device constant
  cudaMemcpyToSymbol(d_NUM_VALS, &NUM_VALS, sizeof(int));

  dim3 blocks(BLOCKS, 1);    /* Number of blocks   */
  dim3 threads(THREADS, 1);  /* Number of threads  */

  CALI_MARK_BEGIN("comp_large");

  for (int i = 0; i < NUM_VALS - 1; ++i) {
    kernel_call++;
    bubble_sort_step<<<blocks, threads>>>(dev_values);
  }

  cudaDeviceSynchronize();
  CALI_MARK_END("comp_large");

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);

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

  CALI_MARK_BEGIN("comp");
  clock_t start, stop;
  CALI_MARK_BEGIN("data_init");
  float *values = (float*)malloc(NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);
  CALI_MARK_END("data_init");
  

  start = clock();
  bubble_sort(values); /* Inplace */
  stop = clock();

  CALI_MARK_END("comp");

  print_elapsed(start, stop);

  size_t size = NUM_VALS * sizeof(float);

  // Calculate metrics based on kernel_call (number of iterations)
  float data_size_gb = (kernel_call * size * 4 * (1e-9)); // Size in GB
  float kernel_execution_time_s = (float)(stop - start) / CLOCKS_PER_SEC; // Kernel execution time in seconds
  float effective_bandwidth_gb_s = (data_size_gb) / kernel_execution_time_s;
  printf("Effective Bandwidth (GB/s): %.6fGB/s\n", effective_bandwidth_gb_s);

  CALI_MARK_BEGIN("correctness_check");
  if(verify(values)){
    printf("sort successful\n");
  }
  else{
    printf("sort unsuccessful\n");
  }
  CALI_MARK_END("correctness_check");

  free(values);

  return 0;
}
