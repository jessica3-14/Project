#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
//#include "helper.cpp"

#define MASTER 0

void genData(int dataSize, int mode, double* test){
    int world_size;
    MPI_Comm_size( MPI_COMM_WORLD , &world_size);
    int world_rank;
    MPI_Comm_rank( MPI_COMM_WORLD , &world_rank);
    int n_local_vals = dataSize/world_size;
    if(mode == 0){
        //randomly sorted data
        for (int i = 0; i<n_local_vals; i++){
            test[i] = rand() % 100000;
        }
    }
    else if(mode == 1){
        //sorted data
        for (int i = 0; i < n_local_vals; i++) {
            test[i] = 100000 / world_size * world_rank + (100000.0 / (dataSize+1)) * i;
        }
    }
    else if(mode == 2){
        //reverse sorted data
        for (int i = 0; i < n_local_vals; i++) {
            test[n_local_vals-i] = 100000 / world_size * world_rank + (100000.0 / (dataSize+1)) * i;
        }
    }
    else{
        //1% noise
        double temp;
        int noise_index1,noise_index2;
        for(int i=0;i<n_local_vals/100;i++){
            noise_index1 = rand()%n_local_vals;
            noise_index2 = rand()%n_local_vals;

            temp=test[noise_index1];
            test[noise_index1]=test[noise_index2];
            test[noise_index2]=temp;
        } 
    }
}

int partnerFind(int phase, int myrank) {
    int partner;
    if (phase % 2 == 0) {
        if (myrank % 2 == 0) {
            partner = myrank + 1;
        } else {
            partner = myrank - 1;
        }
    } else {
        if (myrank % 2 == 0) {
            partner = myrank - 1;
        } else {
            partner = myrank + 1;
        }
    }
    return partner;
}

int verify(double *test, int dataSize){
  for(int i = 0; i < dataSize-1; i++){
    if(test[i] > test[i+1]){
      return -1;
    }
  }
  return 1;
}

int main(int argc, char** argv){
    const char* main = "main";
    const char* data_init = "data_init";
    const char* comm = "comm";
    const char* comm_large = "comm_large";
    const char* comp = "comp";
    const char* comp_large = "comp_large";
    const char* correctness_check = "correctness_check";
    
    cali::ConfigManager mgr;
    mgr.start();

    int my_rank, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    CALI_MARK_BEGIN(main);

    if (numprocs < 2) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int mode = 0;
    int dataSize = 100;

    dataSize = std::atoi(argv[1]);
    mode = std::atoi(argv[2]);

    unsigned long local_data = dataSize / numprocs;
    double* test = new double[local_data];
    double* final_arr = nullptr;

    CALI_MARK_BEGIN(data_init);
    genData(dataSize, mode, test);
    CALI_MARK_END(data_init);

    MPI_Barrier(MPI_COMM_WORLD);

    //bubble_sort(test, numprocs, local_data, my_rank);
    const unsigned long size = local_data * 2;
    auto* temp = new double[local_data];
    auto* data = new double[size];

    for(int i = 0; i < numprocs; i++){
        int partner = partnerFind(i, my_rank);
        if(partner >= numprocs || partner < 0){
            continue;
        }

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        if(my_rank % 2 == 0){
            MPI_Send(test, (double) local_data, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
            MPI_Recv(temp, (double) local_data, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Recv(temp, (double) local_data, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(test, (double) local_data, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
        }
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        std::merge(test, test+local_data, temp, temp+local_data, data);
        std::sort(data, data+size);

        auto half_data = data + local_data;

        if(my_rank < partner){
            std::copy(data, half_data, test);
        }
        else{
            std::copy(half_data, data+size, test);
        }
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
    }

    if (my_rank == MASTER) {
        final_arr = new double[dataSize];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather( test , local_data , MPI_DOUBLE , final_arr , local_data , MPI_DOUBLE , 0 , MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if (my_rank == MASTER) {
        CALI_MARK_BEGIN(correctness_check);
        if (verify(final_arr, dataSize) == 1) {
            printf("Sort successful\n");
        } else {
            printf("Sort unsuccessful\n");
        }
        CALI_MARK_END(correctness_check);

        delete[] final_arr;
        delete[] test;
    }
    CALI_MARK_END(main);
    
    if(my_rank == MASTER){
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Bubblesort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "double");
    adiak::value("SizeOfDatatype", sizeof(double));
    adiak::value("InputSize", dataSize);
    adiak::value("InputType", mode);
    adiak::value("num_procs", numprocs);
    adiak::value("group_num", 12);
    adiak::value("implementation_source", "https://github.com/erenalbayrak/Odd-Even-Sort-mit-MPI/blob/master/implementation/c%2B%2B/OddEvenSort.cpp");

    mgr.stop();
    mgr.flush();
    }

    MPI_Finalize();
    return 0;
}
