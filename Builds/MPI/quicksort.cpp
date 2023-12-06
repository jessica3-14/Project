#include "helper.cpp"
#include <caliper/cali.h>
#include<caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2


int quicksort_step(double* arr, int left, int right){
    double piv = arr[right];
    int i = left - 1;
    
    for(int j = left ; j < right;j++){
        if (arr[j] <= piv){
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    std::swap(arr[i+1], arr[right]);
    return i+1;
}

void quicksort(double* arr, int left, int right){
    if (left < right){
        int piv = quicksort_step(arr, left,right);
        quicksort(arr,left,piv-1);
        quicksort(arr,piv+1, right);
    }
}

int verify(double *test, int dataSize){
  for(int i = 0; i < dataSize; i++){
    if(test[i] > test[i+1]){
      return -1;
    }
  }
  return 1;
}


int main(int argc, char** argv){

int taskid;
int numprocs;
int mtype;
int source;
int destination;
int numworkers;
int rc;

MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

//replace below with comm_large and comm_small

const char* comp = "whole_computation";
const char* master_initialization = "master_initialization";
const char* comm_large = "master_quick";
const char* comm_small = "worker_send";
//const char* worker_receive = "worker_receive";
const char* comp_small = "worker_calculation";

//adiak values
const char* programmingModel = "MPI";
const char* datatype = "double";
int sizeOfDatatype = sizeof(double);
int inputSize = 1000;
const char* inputType = "Random";
int num_procs = numprocs;  // You need to assign the correct value
//int num_threads = 1;  // You need to assign the correct value
//int num_blocks = 1;   // You need to assign the correct value
int group_number = 12; // You need to assign the correct value
const char* implementation_source = "Geeks for Geeks";

//double whole_comp_start_time, whole_comp_end_time, whole_comp_time;
//double master_init_start_time, master_init_end_time, master_init_time;
//double master_quick_start_time, master_quick_end_time, master_quick_time;
//double worker_send_start_time, worker_send_end_time, worker_send_time;
//double worker_receive_start_time, worker_receive_end_time, worker_receive_time;
//double worker_calc_start_time, worker_calc_end_time, worker_calc_time;

MPI_Comm workcom;
MPI_Comm_split(MPI_COMM_WORLD, taskid != 0, taskid, &workcom);

if (numprocs < 2 ) {
  //printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numprocs-1;

//printf("Process %d is sending the array to workers\n", taskid);

//start whole computation
CALI_MARK_BEGIN(comp);
//whole_comp_start_time = MPI_Wtime();

cali::ConfigManager mgr;
mgr.start();#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int	num_procs,             /* number of processes in partition */
	taskid,               /* a process identifier */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	local_size,            /* entries of array sent to each worker */
	avg, extra, offset;    /* used to determine rows sent to each worker */
bool sort = 1;

void swap(int* arr, int i, int j)
{
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

void quicksort(int* arr, int start, int end)
{
    if (end <= 1)
        return;

    int pivot = arr[start + end / 2];

    swap(arr, start, start + end / 2);

    int index = start;

    for (int i = start + 1; i < start + end; i++) {
        if (arr[i] < pivot) {
            // Swap elements if they are less than the pivot
            index++;
            swap(arr, i, index);
        }
    }

    swap(arr, start, index);

    quicksort(arr, start, index - start);
    quicksort(arr, index + 1, start + end - index - 1);
}

int* merge(int* arr1, int n1, int* arr2, int n2)
{
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0;
    int j = 0;
    int k;
 
    while (i < n1 && j < n2)
        result[k++] = arr1[i] < arr2[j] ? arr1[i++] : arr2[j++];
    while (i < n1)
        result[k++] = arr1[i++];
    while (j < n2)
        result[k++] = arr2[j++];
    return result;
}

void genData(int* nums, int size, int mode) {
    avg = floor(size / num_procs);
    extra = size % num_procs;
    if (taskid < extra){
        local_size = avg + 1;
        offset = taskid * avg + taskid;
    }else{
        local_size = avg;
        offset = taskid * avg + extra;
    }


    int* data = (int*)malloc(local_size * sizeof(int));

    if(mode == 0) {
        for(int i = 0; i < local_size; i++) {
            data[i] = rand() % size;
        }
    }
    if(mode == 1) {
        for(int i = 0; i < local_size; i++) {
            data[i] = offset + i;
        }
    }
    if(mode == 2) {
        for(int i = 0; i < local_size; i++) {
            data[i] = size - offset - i - 1;
        }
    }
    if(mode == 3) {
        for(int i = 0; i < local_size; i++) {
            data[i] = offset + i;
        }
        for(int i = 0; i < local_size; i++) {
            if(rand() % 100 == 0) {
                swap(data[i], data[rand() % local_size]);
            }
        }
    }
    

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("mpi_gather");
    MPI_Gather(data, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("mpi_gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    free(data);
}

int verify(int* test, int size) {
    for(int i = 0; i < local_size - 1; i++) {
        int index = i + offset;
        if(test[index] > test[index + 1]) {
            sort = 0;
            return 0;
        }
    }
    return 1;
}


int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    MPI_Status status;
    double time_taken;

    // get MPI info
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // retrieve user input
    int mode = atoi(argv[2]);
    int size = atoi(argv[1]);
    int* nums = (int*)malloc(size * sizeof(int));

    CALI_MARK_BEGIN("data_init");
    genData(nums, size, mode);
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("mpi_bcast");
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("mpi_bcast");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    int chunk_size = (size % num_procs == 0) ? (size / num_procs) : size / (num_procs - 1);

    int* chunk = (int*)malloc(chunk_size * sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("mpi_scatter");
    MPI_Scatter(nums, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("mpi_scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    free(nums);
    nums = NULL;

    int own_chunk_size;

    if (size >= chunk_size * (taskid + 1)) {
        own_chunk_size = chunk_size;
    } else {
        own_chunk_size = size - chunk_size * taskid;
    }
 
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    quicksort(chunk, 0, own_chunk_size);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

     for (int step = 1; step < num_procs; step = 2 * step) {
        
        if (taskid + step < num_procs) {
            int received_chunk_size = (size >= chunk_size * (taskid + 2 * step)) ? (chunk_size * step) : (size - chunk_size * (taskid + step));
            int* chunk_received;
            chunk_received = (int*)malloc( received_chunk_size * sizeof(int));

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_small");
            CALI_MARK_BEGIN("mpi_recv");
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, taskid + step, 0, MPI_COMM_WORLD, &status);
            CALI_MARK_END("mpi_recv");
            CALI_MARK_END("comm_small");
            CALI_MARK_END("comm");

            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_large");
            nums = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);
            CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");

            free(chunk);
            free(chunk_received);
            chunk = nums;
            own_chunk_size = own_chunk_size + received_chunk_size;
        }
        if (taskid % (2 * step) != 0) {
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_small");
            CALI_MARK_BEGIN("mpi_send");
            MPI_Send(chunk, own_chunk_size, MPI_INT, taskid - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END("mpi_send");
            CALI_MARK_END("comm_small");
            CALI_MARK_END("comm");
            break;
        }
    }

    if(taskid != 0) {
        nums = (int*)malloc(size * sizeof(int));
    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("mpi_bcast");
    MPI_Bcast(nums, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("mpi_bcast");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness_check");
    int st = 1;
    int arr = verify(nums, size);
    CALI_MARK_END("correctness_check");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("mpi_reduce");
    MPI_Reduce(&arr, &st, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    CALI_MARK_END("mpi_reduce");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    if(taskid == 0) {

        if(sort == 1) {
            cout << "Sort Successful" << endl;
        }
        else {
            cout << "Sort Unsuccessful" << endl;
        }
    }

    adiak::init(NULL);
    adiak::launchdate();    
    adiak::libraries();     
    adiak::cmdline();       
    adiak::clustername();  
    adiak::value("Algorithm", "QuickSort"); 
    adiak::value("ProgrammingModel", "MPI"); 
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", 
    adiak::value("InputSize", size);
    adiak::value("InputType", mode); 
    adiak::value("num_procs", num_procs); 
    adiak::value("group_num", 12); 
    adiak::value("implementation_source", "Geek for Geeks and Online"); 

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    free(nums);
    
}

double* test = new double[1000];
int chunkSize = 1000 / (numprocs - 1);


if(taskid == MASTER){
 // printf("quick_sort has started with %d tasks.\n", numprocs);
  //printf("****************************************************************\n");
  //start master initialization 
  CALI_MARK_BEGIN(master_initialization);
  //master_init_start_time = MPI_Wtime();

  //genData(1000,1,test);
  for(int i = 0; i < 1000; i++){
    test[i] = rand() % 1000;
  }
  
  //master_init_end_time = MPI_Wtime();
  //master_init_time = master_init_end_time - master_init_start_time;
  CALI_MARK_END(master_initialization);
  //end master initialization
  
  //start master quicksort
  CALI_MARK_BEGIN(comm_large);
  //master_quick_start_time = MPI_Wtime();
  
  mtype = FROM_MASTER;
  //std::cout << "sending array from master to worker" << std::endl;
  destination = 1;
  for(int dest = 1; dest < numprocs; dest++){
    MPI_Send(&chunkSize, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    MPI_Send(&test[(dest-1)*chunkSize], chunkSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
  }
  
  mtype = FROM_WORKER;
  //std::cout << "receiving array from master to worker" << std::endl;
  for(int source = 1; source < numprocs; source++){
    MPI_Recv(&test[(source-1)*chunkSize], chunkSize, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
  }
  
 // master_quick_end_time = MPI_Wtime();
  //master_quick_time = master_quick_end_time - master_quick_start_time;
  CALI_MARK_END(comm_large);
}
//end master quicksort

if(taskid > MASTER){
  //start worker receive
  CALI_MARK_BEGIN(comm_small);
  //worker_receive_start_time = MPI_Wtime();
  
  mtype = FROM_MASTER;
  MPI_Recv(&chunkSize, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
  
  double* localData = new double[chunkSize];
  
  MPI_Recv(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
  
  //worker_receive_end_time = MPI_Wtime();
  //worker_receive_time = worker_receive_end_time - worker_receive_start_time;
  CALI_MARK_END(comm_small);
  //end worker receive
  
  //start worker calculation
  CALI_MARK_BEGIN(comp_small);
  //worker_calc_start_time = MPI_Wtime();
  
  quicksort(localData, 0, chunkSize-1);
  
  //worker_calc_end_time = MPI_Wtime();
  //worker_calc_time = worker_calc_end_time - worker_calc_start_time;
  CALI_MARK_END(comp_small);
  //end worker calculation
  
  //start worker send
  CALI_MARK_BEGIN(comm_small);
  //worker_send_start_time = MPI_Wtime();
  
  mtype = FROM_WORKER;
  MPI_Send(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  
  //worker_send_end_time = MPI_Wtime();
  //worker_send_time = worker_send_end_time - worker_send_start_time;
  CALI_MARK_END(comm_small);
  //end worker send

  delete[] localData;
}

//end whole computation
//whole_comp_end_time = MPI_Wtime();
//whole_comp_time = whole_comp_end_time - whole_comp_start_time;
CALI_MARK_END(comp);

//int ret=0;
//check_sort(test,1000,&ret);
//printf("%d\n",ret);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster



//double worker_receive_time_max, worker_receive_time_min, worker_receive_time_sum, worker_receive_time_avg = 0;
//double worker_send_time_max, worker_send_time_min, worker_send_time_sum, worker_send_time_avg = 0;
//double worker_calc_time_max, worker_calc_time_min, worker_calc_time_sum, worker_calc_time_avg = 0;

//MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
//MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
//MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);

//MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
//MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
//MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);

//MPI_Reduce(&worker_calc_time, &worker_calc_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
//MPI_Reduce(&worker_calc_time, &worker_calc_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
//MPI_Reduce(&worker_calc_time, &worker_calc_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);



if(taskid == 0){
/*
  printf("\n");
  printf("Master times:\n");
  printf("whole computation time: %f\n", whole_comp_time);
  printf("master initialization time: %f\n", master_init_time);
  printf("master quicksort time: %f\n", master_quick_time);
  printf("\n");
  
  adiak::value("MPI_Reduce-whole-computation_time", whole_comp_time);
  adiak::value("MPI_Reduce-master_initialization_time", master_init_time);
  adiak::value("MPI_Reduce-master_quick_time", master_quick_time);
  
  mtype = FROM_WORKER;
  MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_receive_time_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_send_time_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_calc_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_calc_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&worker_calc_time_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
  
  adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
  adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
  adiak::value("MPI_Reduce-worker_receive_time_avg", worker_receive_time_avg);
  adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
  adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
  adiak::value("MPI_Reduce-worker_send_time_avg", worker_send_time_avg);
  adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calc_time_max);
  adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calc_time_min);
  adiak::value("MPI_Reduce-worker_calculation_time_avg", worker_calc_time_avg);
  */
  CALI_MARK_BEGIN("correctness_check");
  if(verify(test, 1000)){
    printf("Sort successful\n");
  }
  else{
    printf("Sort unsuccessful\n");
  }
  CALI_MARK_END("correctness_check");
  adiak::value("Algorithm", "QuickSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source);
}
/*
else if(taskid == 1){

  worker_receive_time_avg = worker_receive_time_sum / (double)numworkers;
  worker_send_time_avg = worker_send_time_sum / (double)numworkers;
  worker_calc_time_avg = worker_calc_time_sum / (double)numworkers;
  
  printf("\n");
  printf("Worker times:\n");
  printf("Min worker receive time: %f\n", worker_receive_time_min);
  printf("Max worker receive time: %f\n", worker_receive_time_max);
  printf("Average worker receive time: %f\n", worker_receive_time_avg);
  //printf("worker receive time: %f\n", worker_receive_time);
  printf("Min worker send time: %f\n", worker_send_time_min);
  printf("Max worker send time: %f\n", worker_send_time_max);
  printf("Average worker send time: %f\n", worker_send_time_avg);
  //printf("worker send time: %f\n", worker_send_time);
  printf("Min worker calculation time: %f\n", worker_calc_time_min);
  printf("Max worker calculation time: %f\n", worker_calc_time_max);
  printf("Average worker calculation time: %f\n", worker_calc_time_avg);
  //printf("worker calculation time: %f\n", worker_calc_time);
  
  mtype = FROM_WORKER;
  MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_receive_time_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_send_time_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_calc_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_calc_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&worker_calc_time_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
}
*/
mgr.stop();
mgr.flush();

MPI_Finalize();
}
