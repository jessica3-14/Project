#include <iostream>
#include <thrust/device_vector.h>
#include "helper.cu"
#include <thrust/host_vector.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <thrust/sort.h>


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

    int blockSize = 256;

    int numBlocks = (numElements + blockSize - 1) / blockSize;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    bucketSort<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()),thrust::raw_pointer_cast(d_sizes.data()), numElements, numBuckets);
CALI_MARK_END(comp_large);
CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
for(int i=0;i<numBuckets;i++){
thrust::sort(thrust::device,d_output.begin()+numElements*i,d_output.begin()+numElements*i+d_sizes[i],thrust::less<double>());
}
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
mgr.stop();
   mgr.flush();

    return 0;

}
