#include <inttypes.h>

#include <stdlib.h>

#include <stdio.h>

#include <math.h>

#include <mpi.h>
#include <caliper/cali.h>

#include<caliper/cali-manager.h>

#include <adiak.hpp>

// Comparison function used by qsort

int compare_dbls(const void* arg1, const void* arg2)

{

    double a1 = *(double *) arg1;

    double a2 = *(double *) arg2;

    if (a1 < a2) return -1;

    else if (a1 == a2) return 0;

    else return 1;

}
// verification function

int verify(double *array, int array_len)
{
   int i;
   for (i=0 ; i < array_len-1 ; i++)
   {
      if (array[i] > array[i+1])
         return -1;
   }
   return 1;
}

void qsort_dbls(double *array, int array_len)
{
    qsort(array, (size_t)array_len, sizeof(double),
          compare_dbls);
}


int main( int argc, char *argv[] )
{
CALI_CXX_MARK_FUNCTION;
   MPI_Init(&argc, &argv);

   int n, i;

   double *input_array;   //the input array

   double *bucketlist;    //this array will contain input elements in order of the processors

                          //e.g elements of process 0 will be stored first, then elements of process 1, and so on

   int *scounts;          //This array will contain the counts of elements each processor will receive

   int *dspls;            //The rel. offsets in bucketlist where the elements of different processes will be stored

   double *local_array;   //This array will contain the elements in each process

   int *bin_elements;     //it will keep track of how many elements have been included in the pth bin

   double *sorted_array;  //final sorted array


int numElements = atoi(argv[1]);
int mode = atoi(argv[2]);

const char* main = "main";

const char* data_init = "data_init";

const char* comm = "comm";

const char* comm_small = "comm_small";

const char* comm_large = "comm_large";

const char* comp = "comp";

const char* comp_large = "comp_large";

const char* comp_small = "comp_small";

const char* correctness_check = "correctness_check";

   int p, rank;

   MPI_Comm_size(MPI_COMM_WORLD, &p);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int numBuckets = p;

cali::ConfigManager mgr;

    mgr.start();

   scounts = new int[p];
   dspls = new int[p];


CALI_MARK_BEGIN(data_init);
   if (rank == 0)
   {

      if (argc == 1)

      {
         fprintf(stderr, "ERROR: Please specify the number of elements.\n");
         exit(1);
      }

      n = atoi(argv[1]);
      input_array = new double[n];

      bucketlist = new double[n];
      bin_elements = new int[p];


      for(i = 0 ; i < n ; i++)
      {
        input_array[i] = ((double) rand()/RAND_MAX);
      }
      for(i = 0 ; i < p ; i++)
      {
         scounts[i] = 0 ;
      }

      for(i = 0 ; i < n ; i++)

      {

         scounts[(int)(input_array[i]/(1.0/p))]++;

      }



      for(i = 0 ; i<p ; i++)

      {

         bin_elements[i] = scounts[i];

      }

      dspls[0] = 0;
      for(i = 0 ; i< p-1 ;i++)
      {
         dspls[i+1] = dspls[i] + scounts[i];
      }


      int bin;
      int pos;
      for(i = 0 ; i < n ; i++)
      {

         bin = (int)(input_array[i]/(1.0/p));

         pos = dspls[bin] + scounts[bin] - bin_elements[bin];

         bucketlist[pos] = input_array[i];

         bin_elements[bin]--;

      }


   

   }
CALI_MARK_END(data_init);


 

CALI_MARK_BEGIN(comm);
CALI_MARK_BEGIN(comm_small);
   MPI_Bcast(scounts, p, MPI_INT, 0, MPI_COMM_WORLD); 

   MPI_Bcast(dspls, p, MPI_INT, 0, MPI_COMM_WORLD); 

   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
CALI_MARK_END(comm_small);



   local_array = new double[scounts[rank]];
   sorted_array = new double[n];

CALI_MARK_BEGIN(comm_large);
   MPI_Scatterv(bucketlist, scounts, dspls, MPI_DOUBLE,
                local_array, scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
CALI_MARK_END(comm_large);
CALI_MARK_END(comm);

CALI_MARK_BEGIN(comp);
CALI_MARK_BEGIN(comp_large);
   qsort_dbls(local_array, scounts[rank]);
CALI_MARK_END(comp_large);
CALI_MARK_END(comp);

CALI_MARK_BEGIN(comm);
CALI_MARK_BEGIN(comm_large);
   MPI_Gatherv(local_array, scounts[rank], MPI_DOUBLE, 
               sorted_array, scounts, dspls, MPI_DOUBLE,0, MPI_COMM_WORLD); 
CALI_MARK_END(comm_large);
CALI_MARK_END(comm);


CALI_MARK_BEGIN(correctness_check);
   if (rank == 0)
   {

      if (verify(sorted_array, n)){
         printf("Successful sort\n");
}
      else{
        printf("Unsuccessful sort\n");
}
CALI_MARK_END(correctness_check);

   }
   MPI_Finalize();
adiak::init(NULL);

adiak::launchdate();    // launch date of the job

adiak::libraries();     // Libraries used

adiak::cmdline();       // Command line used to launch the job

adiak::clustername();   // Name of the cluster

adiak::value("Algorithm", "bucket"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")

adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"

adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)

adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)

adiak::value("InputSize", numElements); // The number of elements in input dataset (1000)

adiak::value("InputType",mode); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"

adiak::value("num_procs", numBuckets); // The number of processors (MPI ranks)

//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
//
////adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
//
adiak::value("group_num", 12); // The number of your group (integer, e.g., 1, 10)
//
adiak::value("implementation_source", "AI");





mgr.stop();

   mgr.flush();
   return 0;
}
