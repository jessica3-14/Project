#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
//#include <adiak.hpp>
#include <cuda.h>
#include <cuda_runtime.h>


int THREADS;
int BLOCKS;
int NUM_VALS;

__device__ int d_NUM_VALS; // Declare as a device constant

const char* sample_sort_step_region = "sample_sort_step";
const char* entire_computation_region = "entire_computation";
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
bool check_sorted(double *arr, int arr_len)
{
   for (int i=0 ; i < arr_len-1 ; i++)
   {
      if (arr[i] > arr[i+1]){
        return false;
      }
   }
   return true;
}

__global__ void sample_sort_step(float *data)
{
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int n =  blockDim.x;
    int j, temp;

  for (i = 0; i < n; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (data[j] > data[j + 1]) {
                temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

void sample_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**)&dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  // Copy NUM_VALS to device constant
  cudaMemcpyToSymbol(d_NUM_VALS, &NUM_VALS, sizeof(int));

  dim3 blocks(BLOCKS, 1);    /* Number of blocks   */
  dim3 threads(THREADS, 1);  /* Number of threads  */

  CALI_MARK_BEGIN(sample_sort_step_region);

  //for (int i = 0; i < NUM_VALS - 1; ++i) {
   // kernel_call++;
  //  sample_sort_step<<<blocks, threads>>>(dev_values);
 // }
    sample_sort_step<<<blocks, threads>>>(dev_values);

  cudaDeviceSynchronize();
  CALI_MARK_END(sample_sort_step_region);

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

  clock_t start, stop;

    float *values = (float*)malloc(NUM_VALS * sizeof(float));   //array of values

    array_fill(values, NUM_VALS);

    const char* algorithm = "SampleSort";
    const char* programmingModel = "CUDA";
    const char* datatype = "float";
    unsigned int sizeOfDatatype = sizeof(float);
    int inputSize = NUM_VALS;
    const char* inputType = "Random";
    const char* group_number = "12";
    const char* implementation_source = "AI";
    int num_procs = THREADS;
    int num_threads = THREADS;
    int num_blocks = BLOCKS;


/*    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster

*/
  CALI_MARK_BEGIN(entire_computation_region);

  start = clock();
  sample_sort(values); /* Inplace */
  stop = clock();

  CALI_MARK_END(entire_computation_region);

  print_elapsed(start, stop);

  size_t size = NUM_VALS * sizeof(float);

  // Calculate metrics based on kernel_call (number of iterations)
  float data_size_gb = (kernel_call * size * 4 * (1e-9)); // Size in GB
  float kernel_execution_time_s = (float)(stop - start) / CLOCKS_PER_SEC; // Kernel execution time in seconds
  float effective_bandwidth_gb_s = (data_size_gb) / kernel_execution_time_s;
  printf("Effective Bandwidth (GB/s): %.6fGB/s\n", effective_bandwidth_gb_s);



/*  if (check_sorted(values, NUM_VALS))
         printf("Sorted with Sample Sort\n");
      else
        printf("Unsuccessful Sort\n");

    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

*/






  free(values);

  return 0;
}

