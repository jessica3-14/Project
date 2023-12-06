#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include "helper.cu"
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

__device__ int d_NUM_VALS;
const char* comp = "comp";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp_small = "comp_small";
//const char* bitonic_sort_step_region = "bitonic_sort_step";
//const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
//const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

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
int verify(float *values){
  for(int i = 0; i < sizeof(values)-1; i++){
    if(values[i] > values[i+1]){
      return -1;
    }
  }
  return 1;
}
__global__ int quick_sort_step(float* dev_values, int l, int h) {
  float piv = dev_values[(h+l)/2];
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  
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
  return left;
}

/**
 * Inplace bitonic sort using CUDA.
 */
void quick_sort(float* values, int low, int high) {
  float* dev_values;
  size_t size = (high - low + 1) * sizeof(float);
  
  //CALI_MARK_BEGIN(comm);

  cudaMalloc((void**)&dev_values, size);
  
  // MEM COPY FROM HOST TO DEVICE
  //CALI_MARK_BEGIN(comm_large);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  //CALI_MARK_END(comm_large);
  
  cudaMemcpyToSymbol(d_NUM_VALS, &NUM_VALS, sizeof(int));
  
  //CALI_MARK_END(comm);

  dim3 blocks(BLOCKS, 1);    /* Number of blocks   */
  dim3 threads(THREADS, 1);  /* Number of threads  */

  // Major step
  //CALI_MARK_BEGIN(comp);
  //CALI_MARK_BEGIN(comp_small);
  
  quick_sort_step<<<blocks, threads>>>(dev_values, 0, size - 1);
  
  //cudaDeviceSynchronize();
  
  //CALI_MARK_END(comp_small);
  //CALI_MARK_END(comp);

  // MEM COPY FROM DEVICE TO HOST
  //CALI_MARK_BEGIN("comm");
  //CALI_MARK_BEGIN("comm_large");
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  //CALI_MARK_END("comm_large");
  //CALI_MARK_END("comm");
  
  cudaFree(dev_values);
}

int main(int argc, char* argv[]) {
  CALI_MARK_BEGIN("main");
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  int mode = atoi(argv[3]);
  BLOCKS = NUM_VALS / THREADS;
  
  size_t size = NUM_VALS * sizeof(float);


  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  cali::ConfigManager mgr;
  mgr.start();
  
  clock_t start, stop;
  
  CALI_MARK_BEGIN("data_init");
  double *tvals = (double *)malloc(NUM_VALS * sizeof(double));
  genData(NUM_VALS, mode, tvals, THREADS);
  float *values = (float*)malloc(NUM_VALS * sizeof(float));
  //array_fill(values, NUM_VALS);
  for (int i = 0; i < NUM_VALS;i++){
    values[i] = (float)tvals[i];
  }
  CALI_MARK_END("data_init");
  free(tvals);
  
    float *dev_values;
    cudaMalloc((void**) &dev_values, size);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    //CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    //CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    /*
    for (int i = 0; i < NUM_VALS; i++) {
        quicksort<<<BLOCKS, THREADS>>>(dev_values, NUM_VALS);
    }
    */
    quick_sort(values, 0, NUM_VALS - 1);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
  
  
  /*
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("large_comp");
  start = clock();
  quick_sort(values, 0, NUM_VALS - 1);
  stop = clock();
  CALI_MARK_END("large_comp");
  CALI_MARK_END("comp");
  */
  
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
  
  adiak::init(NULL);
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Quicksort");
  adiak::value("ProgrammingModel", "CUDA");
  adiak::value("Datatype", "float");
  adiak::value("SizeOfDatatype", 8);
  adiak::value("InputSize", NUM_VALS);
  adiak::value("InputType", mode);
  adiak::value("num_threads", THREADS);
  adiak::value("group_num", 12);
  adiak::value("implementation_source", "https://www.geeksforgeeks.org/quick-sort/");

 
  mgr.stop();
  mgr.flush();
}
