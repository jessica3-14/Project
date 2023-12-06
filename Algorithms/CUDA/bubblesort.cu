#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include<caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper.cu"

int THREADS;
int BLOCKS;
int NUM_VALS;

__device__ int d_NUM_VALS; // Declare as a device constant

int kernel_call;

const char* comm = "comm";
const char* comp = "comp";
const char* comm_large = "comm_large";

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

__global__ void bubble_sort(float *dev_values, int size)
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < size) {
    for (int i = 0; i < size - tid - 1; ++i) {
      if (dev_values[i] > dev_values[i + 1]) {
             float temp = dev_values[i];
             dev_values[i] = dev_values[i + 1];
             dev_values[i + 1] = temp;
      }
    }
  }
}


int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN("main");
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  int mode = atoi(argv[3]);
  BLOCKS = (NUM_VALS + THREADS -1) / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
  
  cali::ConfigManager mgr;
  mgr.start();

  
  clock_t start, stop;
  CALI_MARK_BEGIN("data_init");
  double *temp_values = (double *)malloc(NUM_VALS * sizeof(double));
  genData(NUM_VALS, mode, temp_values, THREADS);
  float *values = (float*)malloc(NUM_VALS * sizeof(float));
  //array_fill(values, NUM_VALS);
  for(int i = 0; i < NUM_VALS; i++){
    values[i] = (float)temp_values[i];
  }
  CALI_MARK_END("data_init");
  free(temp_values);
  
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);
  
  CALI_MARK_BEGIN(comm);

  cudaMalloc((void**)&dev_values, size);
  
  CALI_MARK_BEGIN(comm_large);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END(comm_large);

  // Copy NUM_VALS to device constant
  cudaMemcpyToSymbol(d_NUM_VALS, &NUM_VALS, sizeof(int));
  
  CALI_MARK_END(comm);
  
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN("large_comp");
  start = clock();
  bubble_sort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS); /* Inplace */
  cudaDeviceSynchronize();
  stop = clock();
  CALI_MARK_END("large_comp");
  CALI_MARK_END(comp);
  
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  cudaFree(dev_values);

  print_elapsed(start, stop);

  size = NUM_VALS * sizeof(float);
  
  adiak::init(NULL);
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Bubblesort");
  adiak::value("ProgrammingModel", "CUDA");
  adiak::value("Datatype", "float");
  adiak::value("SizeOfDatatype", sizeof(double));
  adiak::value("InputSize", NUM_VALS);
  adiak::value("InputType", mode);
  adiak::value("num_threads", THREADS);
  adiak::value("group_num", 12);
  adiak::value("implementation_source", "https://www.geeksforgeeks.org/bubble-sort/");

  CALI_MARK_BEGIN("correctness_check");
  if(verify(values)){
    printf("sort successful\n");
  }
  else{
    printf("sort unsuccessful\n");
  }
  CALI_MARK_END("correctness_check");

  free(values);
  CALI_MARK_END("main");
  
  mgr.stop();
  mgr.flush();

  return 0;
}
