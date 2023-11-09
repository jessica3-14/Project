#include "helper.cpp"
#include <caliper/cali.h>
#include<caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int compare_dbls(const void* arg1, const void* arg2)
{
    double a1 = *(double *) arg1;
    double a2 = *(double *) arg2;
    if (a1 < a2) return -1;
    else if (a1 == a2) return 0;
    else return 1;
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

void qsort_dbls(double *array, int array_len)
{
    qsort(array, (size_t)array_len, sizeof(double),
          compare_dbls);
}



int main(int argc, char** argv){

 

   MPI_Init(&argc, &argv);

   int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;  //harcoded total array size
    int local_N = N / size;  // Number of elements per process
    double* local_data = (double*)malloc(local_N * sizeof(double));
    // now generate random values for local data for sorting
    //CALI_MARK_BEGIN(data_init);

    for (int i = 0; i < local_N; i++) {
    local_data[i] = (double)rand() / RAND_MAX; //populating processes with random values
    }
   //CALI_MARK_END(data_init);

    const char* algorithm = "SampleSort";
    const char* programmingModel = "MPI";
    const char* datatype = "double";
    unsigned int sizeOfDatatype = sizeof(double);
    const char* inputSize = "1000";
    const char* inputType = "Random";
    const char* group_number = "12";
    const char* implementation_source = "AI";
    int num_procs = rank;
    const char* num_threads = "";
    const char* num_blocks = "";
   
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster





    //CALI_MARK_BEGIN(comp_large);
    qsort(local_data, local_N, sizeof(double), compare_dbls);   // Each process sorts its data locally
    double* splitters = (double*)malloc(size * sizeof(double));
    int idx = local_N / 2; 
    MPI_Gather(&local_data[idx], 1, MPI_DOUBLE, splitters, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        qsort(splitters, size, sizeof(double), compare_dbls);           // sorting the splitters
        double* global_splitters = (double*)malloc((size - 1) * sizeof(double));
        for (int i = 0; i < size - 1; ++i) {
            global_splitters[i] = splitters[(i + 1) * size / size]; //to get global splitters
        }
        free(splitters);
        splitters = global_splitters;
    }   //choose global splitters
    // Partition based on the splitters


    //CALI_MARK_END(comp_large);


    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    int i, j = 0;

    //CALI_MARK_BEGIN(comp_small);
    for (i = 0; i < size - 1; ++i) {
        while (j < local_N && local_data[j] < splitters[i]) ++j;
        sendcounts[i] = j;
        displs[i] = j;
    }
    sendcounts[size - 1] = local_N - j;
    displs[size - 1] = j;
    //CALI_MARK_END(comp_small);

    //CALI_MARK_BEGIN(comm_small);
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* recvdispls = (int*)malloc(size * sizeof(int));
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(displs, 1, MPI_INT, recvdispls, 1, MPI_INT, MPI_COMM_WORLD);
    double* new_data = (double*)malloc(N * sizeof(double)); // exchange data given partitions
    //CALI_MARK_END(comm_small);


    //CALI_MARK_BEGIN(comm_large);
    MPI_Alltoallv(local_data, sendcounts, displs, MPI_DOUBLE, new_data, recvcounts, recvdispls, MPI_DOUBLE, MPI_COMM_WORLD);
    //CALI_MARK_END(comm_large);


    //CALI_MARK_BEGIN(comp_large);
    qsort(new_data, N, sizeof(double), compare_dbls);   //Sort received data
    //CALI_MARK_END(comp_large);

    if (rank == 0)
   {
      //CALI_MARK_BEGIN(correctness_check);
      if (check_sorted(new_data, N))
         printf("Sorted with Sample Sort\n");
      else
        printf("Unsuccessful Sort\n");
      //CALI_MARK_END(correctness_check);

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

   }


    MPI_Finalize();
}