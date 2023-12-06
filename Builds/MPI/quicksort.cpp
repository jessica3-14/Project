#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int	num_procs,             
	taskid,               
	source,                
	dest,                  
	mtype,                 
	local_size,            
	avg, extra, offset;    
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
