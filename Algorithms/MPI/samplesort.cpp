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
#include<unistd.h>
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

double* genData(int DATA_SIZE, int mode) {
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int n_local_vals = DATA_SIZE/world_size;


    double* data_arr = new double[n_local_vals];
    // creating new data
  

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
    
    //MPI_Allgather(data_arr,n_local_vals,MPI_DOUBLE,final_arr,n_local_vals,MPI_DOUBLE,MPI_COMM_WORLD);
   return data_arr;
}



int main(int argc, char** argv){
 
cali::ConfigManager mgr;
mgr.start();

double whole_computation_time;
double whole_computation_starttime;
double whole_computation_endtime;
double maybe_all=0;

    int N = atoi(argv[1]);  // input of size
    int array_mode= atoi(argv[2]);

   MPI_Init(&argc, &argv);

   int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    
     //n = atoi(argv[1]);

    int local_N = N / size;  // Number of elements per process
    double* local_data = (double*)malloc(local_N * sizeof(double));
    // now generate random values for local data for sorting
    
    const char* algorithm = "SampleSort";
    const char* programmingModel = "MPI";
    const char* datatype = "double";
    unsigned int sizeOfDatatype = sizeof(double);
    int inputSize = N;
   const char* inputType = "Random";

    if (array_mode == 0){
        inputType = "Random";
    }else if (array_mode ==1){
      inputType = "Sorted";
    }else if (array_mode ==2){
      inputType = "reverse sorted";
    }else{
      inputType = "1 percent noise";
    }

 
    const char* group_number = "12";
    const char* implementation_source = "AI";
    int num_procs = size;
    const char* num_threads = "";
    const char* num_blocks = "";
    
    const char* correctness_check = "correctness_check";
    //const char* master_initialization = "master_initialization";
    const char* comm_large = "comm_large";
    const char* comm_small = "comm_small";
    const char* comp_small = "comp_small";
    const char* comp_large = "comp_large";
    const char* comp = "comp";
    const char* comm = "comm";
    const char* data_init = "data_init";
    const char* whole_comp_time= "whole_comp_time";





   

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster


    


    
    
    

    
    CALI_MARK_BEGIN(data_init);

    //for (int i = 0; i < local_N; i++) {
  //  local_data[i] = (double)rand() / RAND_MAX; //populating processes with random values
   // }
    local_data = genData(N,array_mode);

   CALI_MARK_END(data_init);

  
  CALI_MARK_BEGIN(whole_comp_time);

  whole_computation_starttime= MPI_Wtime();


    CALI_MARK_BEGIN(comp);
    qsort(local_data, local_N, sizeof(double), compare_dbls);   // Each process sorts its data locally

      CALI_MARK_END(comp);
    // take out later
 //   printf("Data in this array:%d\n", rank);
  //  for (int i = 0; i < local_N; ++i) {
   //   printf("%f ", local_data[i]);
   // }
     
  
    // take this out later
  //  int nuts= check_sorted(local_data, local_N);



  //  printf("Checking if sort worked : %d\n", nuts);
    double* splitters = (double*)malloc(size * sizeof(double)*(size-1)); // we have splitters = processes*(processes-1), since it is the combination of all splitters

    if (rank==0){
 //     splitters = (double*)malloc(size * sizeof(double)*(size-1)); // we have splitters = size, one for each processor
 //double* sorted_array = (double*)malloc(N * sizeof(double));
    }
   

    int mod_n = ceil(float(local_N)/(float(size)));
    double* sample_splitter_local = (double*)malloc((size-1) * sizeof(double)); //each local splitter has size of (processes-1)
    CALI_MARK_BEGIN(comp);
    for (int i=1; i< size; i++){
      sample_splitter_local[i-1]= local_data[mod_n*i-1];

      //MPI_Gather(&local_data[mod_n*i], 1, MPI_DOUBLE, splitters+ (i-1)*(size-1), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    }

          CALI_MARK_END(comp);
    if (rank==0){
  printf(" Local splitters made\n");
  }
CALI_MARK_BEGIN(comm);
    MPI_Gather(sample_splitter_local, size-1, MPI_DOUBLE, splitters, size-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     //Gathered splitters from local processes

     //free(sample_splitter_local);
          CALI_MARK_END(comm);

        double* global_splitters = (double*)malloc((size - 1) * sizeof(double)); // want size-1 global splitters

    if (rank == 0) {
        printf("Sorting array size of : %d\n", N);
        printf("With %d processes  \n", size);
        printf("With %d mode  \n", array_mode);
         // splitters are correct


      CALI_MARK_BEGIN(comp);
        qsort(splitters, size*(size-1), sizeof(double), compare_dbls);           // sorting the splitters

        int mod_glob = ceil(float((size-1)*size)/float(size));  //size-1 *size is size of splitter, size-1 since we want size-1 glob_splitters

        for (int i = 1; i < size; ++i) {
            global_splitters[i-1] = splitters[mod_glob*i-1]; //to get global splitters
        }
          CALI_MARK_END(comp);
      //   printf("With Splitters array of : \n ");
      //           for (int i = 0; i < size*(size-1); ++i) {
     //     printf("%f ", splitters[i]);
    //    }
     //   free(splitters);
     //   printf("\n With Global Splitters array of: \n ");  // splitters are correct
      //  for (int i = 0; i < (size-1); ++i) {
     //     printf("%f ", global_splitters[i]);
    //    }

    }   //choose global splitters //end if rank 0

       CALI_MARK_BEGIN(comm);
    MPI_Bcast(global_splitters, size-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   CALI_MARK_END(comm);

      if (rank==0){
    printf(" Splitters broadcasted to all processes\n");
      }

    // We want each process to do one bucket. //Remember, each process still has its own randomly generated data
    //first locally place correct element into respective bucket. then after all elements are placed in respective buckets, 
    //each process should sort a bucket.

  


  //  int* sendcounts = (int*)malloc(size * sizeof(int));
    //int* displs = (int*)malloc(size * sizeof(int));
    
    int numBuckets= size;
  std::vector <double> globalbucketList[numBuckets];
   std::vector <double> bucketList[numBuckets];
    
    
    bool added;

   // printf("\nHELPPPPP :");
    //    for (int i = 0; i < (size-1); ++i) {
    //      printf("%f ", global_splitters[i]);
    //    }


  CALI_MARK_BEGIN(comp);
 
      
    for (int i=0; i<local_N; i++){  //looping through original data
      added= false;
      for (int j=0;j<size-1;j++){ //size = 2
        if (local_data[i]<global_splitters[j]){ //if local data is less than first splitter, then
      //   printf("added %f ", local_data[i]);
          bucketList[j].push_back(local_data[i]);  //add to bucket if splitter value>local data[i]
          added= true;
          break;
        }   // break once added

      }
      if (added==false){  //then we havent added data to a bucket yet, so just add it to last remaining bucket
        //printf(" added to last : ");
        bucketList[size-1].push_back(local_data[i]);
  
      }
    }

  //  free(local_data);
   // free(global_splitters);

    CALI_MARK_END(comp);

    if (rank==0){
    printf(" Global buckets created and values stored\n");
    }
    
  double* sorted_array = (double*)malloc(N * sizeof(double));

 



    CALI_MARK_BEGIN(comm);

    for (int k=0; k<size+1; k++){        //need since we have bucketlist, but we want individual buckets

    
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int bucketListSize = static_cast<int>(bucketList[k].size());

      MPI_Gather(&bucketListSize, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

     // printf(" Gave \n"); we have given main process information about how many elements are in each bucket

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    
  
        // Calculate total size of the global vector
        int totalSize = displs[size - 1] + sendcounts[size - 1];
        globalbucketList[k].resize(totalSize);
    }
    MPI_Gatherv(bucketList[k].data(), bucketList[k].size(), MPI_DOUBLE,
                globalbucketList[k].data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    printf(" Gathered all buckets and their respective data to the main process \n");

if (rank == 0) {  
  //      printf(" \nGlobal Vector:");
  //      for (int asd= 0; asd<globalbucketList[k].size(); asd++){
   //       printf("%f ", globalbucketList[k].at(asd));
  //      }
        printf("Attempting to send data to processes");
   
        int globalbucketsize= static_cast<int>(globalbucketList[k].size());
        MPI_Send(&globalbucketsize, 1, MPI_INT, k, 0, MPI_COMM_WORLD);

        MPI_Send(globalbucketList[k].data(), static_cast<int>(globalbucketList[k].size()), MPI_DOUBLE, k, 0, MPI_COMM_WORLD);
        printf(" Sent bucket to other processes for sorting \n");
    
    
    
    
    
    
    }

  

    
  }// now data has been sent


CALI_MARK_END(comm);




    //basically each process sorts array.
  for (int i=0; i<size; i++){
    if (rank== i){
       int size_of_data;

        CALI_MARK_BEGIN(comm);
      MPI_Recv(&size_of_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double* received_data = (double*)malloc(size_of_data * sizeof(double)); // frieed

      MPI_Recv(received_data, size_of_data, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    CALI_MARK_END(comm);

       CALI_MARK_BEGIN(comp);
      qsort(received_data, size_of_data, sizeof(double), compare_dbls); //sorting bucket locally

        if (rank==0){
      printf(" Process bucket quicksort completed\n");
        }
       CALI_MARK_END(comp);

  //    printf(" \nSorted global vector:\n");
 //     for (int asd= 0; asd<size_of_data; asd++){
   //       printf("%f ", received_data[asd]);
  //      }
      //sorted correctly, now send to process 0
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    //int bucketListSize = static_cast<int>(bucketList[k].size());
      CALI_MARK_BEGIN(comm);
      MPI_Gather(&size_of_data, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
      
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

      
          MPI_Gatherv(received_data, size_of_data, MPI_DOUBLE,
                sorted_array, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD); //double* sorted_array = (double*)malloc(N * sizeof(double));
        //free(received_data);
         CALI_MARK_END(comm);
    }
  }
   
  // now we have correct global bucket list values in rank 0. Send these to correct processors for local sorting, and then return them
  // in correct order to process 0.

  CALI_MARK_END(whole_comp_time);

  whole_computation_endtime= MPI_Wtime();
    
  whole_computation_time= whole_computation_endtime-whole_computation_starttime;

    if (rank == 0)
   {
    //printf("Data in new_data array:\n");
   

    //MPI_Gatherv(new_data, local_N, MPI_DOUBLE, sorted_array, recvcounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // new data is correct, local_N is correct, mpi double is correct, sorted_array is correct,

     // printf("\n Computation time is \n");
    //  printf("Array size is: %f ", N);   
     // printf("Processes: %f ", size);   

      printf("Computation time is: %f ", whole_computation_time);   


      CALI_MARK_BEGIN(correctness_check);
      if (check_sorted(sorted_array, N))
         printf("\n Sorted with Sample Sort\n");
      else
        printf("\n Unsuccessful Sort\n");
      CALI_MARK_END(correctness_check);

     // free(sorted_array);
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
    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

   }

    

   mgr.stop();
    mgr.flush();
    MPI_Finalize();
}