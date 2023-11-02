#include <mpi.h>
#include <stdio.h>
#include <random>
#include <cfloat>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    if(argc < 2){
	printf("Please enter a size of data");
	return 0;
}
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int DATA_SIZE = atoi(argv[1]);
    int n_local_vals = DATA_SIZE/world_size;

    double* data_arr = new double[n_local_vals];
    double* final_arr = NULL;

    std::uniform_real_distribution<double> unif(0, 100000);
    std::default_random_engine re;
    //printf("initialized random engine\n");
    for (int i = 0; i<n_local_vals; i++){
        data_arr[i] = unif(re);
    }
  //  printf("generated data");
    if(world_rank==0){
        final_arr = new double[DATA_SIZE];
    }
    MPI_Gather(data_arr,n_local_vals,MPI_DOUBLE,final_arr,n_local_vals,MPI_DOUBLE,0,MPI_COMM_WORLD);
//printf("gathered data");    

   

    // Finalize the MPI environment.
    MPI_Finalize();
}
