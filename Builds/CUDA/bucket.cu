#include <iostream>
#include <thrust/device_vector.h>
#include "helper.cu"
#include <thrust/host_vector.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <thrust/sort.h>
#include <adiak.hpp>
#include <chrono>

__global__ void bucketSort(double* input, double* output,int* d_sizes, int numElements, int numBuckets) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements) {

        // Determine which bucket the element belongs to
        int bucketIdx = input[tid] * numBuckets / 100000 ;
        // Store the element in the corresponding bucket
	int pos = atomicAdd(&d_sizes[bucketIdx],1);
        output[bucketIdx*numElements+pos] = input[tid];
    }

}



int main(int argc, char** argv) {
const char* main = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* correctness_check = "correctness_check";

    int numElements = atoi(argv[2]);
    int numBuckets = atoi(argv[1]);
    int mode = atoi(argv[3]);

float gpu_time=0;
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN(main);
    CALI_MARK_BEGIN(data_init);
    thrust::host_vector<double> h_input(numElements,0);

    genData(numElements,mode,thrust::raw_pointer_cast(h_input.data()));
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    thrust::device_vector<double> d_input = h_input;
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    thrust::device_vector<double> d_output(numElements*numBuckets);

    thrust::device_vector<int> d_sizes(numBuckets,0);


    // Launch the kernel to distribute elements into buckets

    int blockSize = numBuckets;

    int numBlocks = (numElements + blockSize - 1) / blockSize;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
auto start = std::chrono::high_resolution_clock::now();
    bucketSort<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()),thrust::raw_pointer_cast(d_sizes.data()), numElements, numBuckets);
    cudaDeviceSynchronize();
auto stop = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
gpu_time+= duration.count() /1e6;
CALI_MARK_END(comp_large);
CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
start = std::chrono::high_resolution_clock::now();
for(int i=0;i<numBuckets;i++){
thrust::sort(thrust::device,d_output.begin()+numElements*i,d_output.begin()+numElements*i+d_sizes[i],thrust::less<double>());
}
    cudaDeviceSynchronize();
stop = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
gpu_time+= duration.count() /1e6;
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Copy the sorted data back to the host
CALI_MARK_BEGIN(comm);
CALI_MARK_BEGIN(comm_small);
double* h_output = new double[numElements];
int* h_sizes = new int[numBuckets];
cudaMemcpy(h_sizes,thrust::raw_pointer_cast(d_sizes.data()),numBuckets*sizeof(int),cudaMemcpyDeviceToHost);
CALI_MARK_END(comm_small);
int offset=0;
CALI_MARK_BEGIN(comm_large);
for(int i=0;i<numBuckets;i++){
cudaMemcpy(h_output+offset,thrust::raw_pointer_cast(d_output.data())+numElements*i,h_sizes[i]*sizeof(double),cudaMemcpyDeviceToHost);
offset+=h_sizes[i];
}
CALI_MARK_END(comm_large);
CALI_MARK_END(comm);
    // Print the sorted array



CALI_MARK_BEGIN(correctness_check);
    for (int i = 0; i < numElements-1; i++) {
        if(h_output[i]>h_output[i+1]){
	    printf("Error correctness check failed");
	}

    }
CALI_MARK_END(correctness_check);
CALI_MARK_END(main);

adiak::init(NULL);

adiak::launchdate();    // launch date of the job

adiak::libraries();     // Libraries used

adiak::cmdline();       // Command line used to launch the job

adiak::clustername();   // Name of the cluster

adiak::value("Algorithm", "bucket"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")

adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"

adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)

adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)

adiak::value("InputSize", numElements); // The number of elements in input dataset (1000)

adiak::value("InputType",mode); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"

adiak::value("num_procs", numBuckets); // The number of processors (MPI ranks)
adiak::value("gpu_time", gpu_time);
//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads

//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 

adiak::value("group_num", 12); // The number of your group (integer, e.g., 1, 10)

adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").




mgr.stop();
   mgr.flush();

    return 0;

}
