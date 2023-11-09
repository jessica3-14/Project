#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
//#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

float random_float() {
  return (float)rand() / (float)RAND_MAX;
}

void array_fill(float* arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void quick_sort_step(float* dev_values, int l, int h) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (i >= l && i <= h) {
    float piv = dev_values[l];
    int left = l;
    int right = h;
    
    while (left <= right) {
      while (left <= h && dev_values[left] <= piv) {
        left++;
      }
      while (right >= l && dev_values[right] > piv) {
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
}

/**
 * Inplace bitonic sort using CUDA.
 */
void quick_sort(float* values, int low, int high) {
  float* dev_values;
  size_t size = (high - low + 1) * sizeof(float);

  cudaMalloc((void**)&dev_values, size);
  
  // MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END(cudaMemcpy_host_to_device);

  dim3 blocks(BLOCKS, 1);    /* Number of blocks   */
  dim3 threads(THREADS, 1);  /* Number of threads  */

  // Major step
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  quick_sort_step<<<blocks, threads>>>(dev_values, 0, size - 1);
  cudaDeviceSynchronize();
  CALI_MARK_END(bitonic_sort_step_region);

  // MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_device_to_host);
  cudaFree(dev_values);
}

int main(int argc, char* argv[]) {
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  cali::ConfigManager mgr;
  mgr.start();
  float* values = (float*)malloc(NUM_VALS * sizeof(float));

  clock_t start, stop;
  start = clock();
  quick_sort(values, 0, NUM_VALS - 1);
  stop = clock();

 // print_elapsed(start, stop);

  mgr.stop();
  mgr.flush();
}
