#include <mpi.h>
#include <stdio.h>
#include <random>
#include <cfloat>

void genData(int DATA_SIZE, int mode, double* final_arr) {
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n_local_vals = DATA_SIZE/world_size;

    double* data_arr = new double[n_local_vals];

    std::default_random_engine re;
    //printf("initialized random engine\n");
if(mode==0){ 
    // total random   
    //std::uniform_real_distribution<double> unif(0, 100000);
    for (int i = 0; i<n_local_vals; i++){
        data_arr[i] = rand() % n_local_vals;
    }
}else if(mode==2){
    // reverse sorted 
    for (int i = 0; i < n_local_vals; i++) {
      data_arr[n_local_vals-i] = 100000 / world_size * world_rank + (100000.0 / (DATA_SIZE+1)) * i;
    }
}else{
    // sorted data
    for (int i = 0; i < n_local_vals; i++) {
      data_arr[i] = 100000 / world_size * world_rank + (100000.0 / (DATA_SIZE+1)) * i;
    }
  if(mode==3){
  // adding 1% noise
  double temp;
  int noise_index1,noise_index2;
  for(int i=0;i<n_local_vals/100;i++){
    noise_index1 = rand()%n_local_vals;
    noise_index2 = rand()%n_local_vals;

    temp=data_arr[noise_index1];
    data_arr[noise_index1]=data_arr[noise_index2];
    data_arr[noise_index2]=temp;
  }  
}
}
    
    MPI_Allgather(data_arr,n_local_vals,MPI_DOUBLE,final_arr,n_local_vals,MPI_DOUBLE,MPI_COMM_WORLD);
   
}


void check_sort(double* data, int data_size, int* retval){

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_bool=0;
    int n_local_vals = data_size/world_size;
    for (int i = n_local_vals*world_rank; i < n_local_vals*(world_rank+1); i++){
        if(i==data_size){break;}
        if(data[i] > data[i+1]){ 
            local_bool=1;
        }
    }
    MPI_Allreduce(&local_bool,retval,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);


}
