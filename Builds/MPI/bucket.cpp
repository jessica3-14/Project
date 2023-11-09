
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "helper.cpp"



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



// Sort the array in place
void qsort_dbls(double *array, int array_len)
{
    qsort(array, (size_t)array_len, sizeof(double),
          compare_dbls);
}



int main( int argc, char *argv[] )
{
   
   MPI_Init(&argc, &argv);

   int n, i;
   double *local_input_array;   //The input array for each processor
   double *local_bucketlist;    //This array will contain input elements in order of the processors for each processor
                                //E.g elements of process 0 will be stored first, then elements of process 1, and so on
   int *local_sscounts;         //This array will contain the counts of elements each processor will send to others
   int *local_rscounts;         //This array will contain the counts of elements each processor will receive from others
   int *local_sdspls;           //The offsets in bucketlist where the elements of different processes will be stored - send side
   int *local_rdspls;           //The offsets in local_array where the elements of different processes will be stored - receive side
   double *local_array;         //This array will contain the corrsponding elements in each process
   int *local_bin_elements;     //It will keep track of how many elements have been included in the pth bin
   int *local_array_sizes;      //number of ellements each process has to sort
   int *fdspls;                 //final offset in sorted_array
   double *sorted_array;        //final sorted array




   int p, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &p);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   local_sscounts = new int[p];
   local_rscounts = new int[p];
   local_sdspls = new int[p];
   local_rdspls = new int[p];
   fdspls = new int[p];

   
  n = atoi(argv[1]);
  int mode = atoi(argv[2]); 
      
   local_input_array = new double[n/p];
   //CALI_MARK_BEGIN(data_init);
   genData(n/p,mode,local_input_array);
   //CALI_MARK_END(data_init);
   local_bucketlist = new double[n/p];

   local_bin_elements = new int[p];
   local_array_sizes = new int[p];

   
   //CALI_MARK_BEGIN(comp_large);
   for(i = 0 ; i < p ; i++)
   {
      local_sscounts[i] = 0 ;
   }
   
   
   for(i = 0 ; i < n/p ; i++)
   {
      local_sscounts[(int)(local_input_array[i]/(1.0/p))]++;
   }
   
   for(i = 0 ; i<p ; i++)
   {
      local_bin_elements[i] = local_sscounts[i];
   }
    
   local_sdspls[0] = 0;
   for(i = 0 ; i< p-1 ;i++)
   {
      local_sdspls[i+1] = local_sdspls[i] + local_sscounts[i];
   }
        
   int bin;
   int pos;
   for(i = 0 ; i < n/p ; i++)
   {
      bin = (int)(local_input_array[i]/(1.0/p));
      pos = local_sdspls[bin] + local_sscounts[bin] - local_bin_elements[bin];
      local_bucketlist[pos] = local_input_array[i];
      local_bin_elements[bin]--;
   }
   //CALI_MARK_END(comp_large);

   //CALI_MARK_BEGIN(comm_small);
   MPI_Allreduce(local_sscounts, local_array_sizes, p, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   local_array = new double[local_array_sizes[rank]];

   MPI_Alltoall(local_sscounts,1, MPI_INT,local_rscounts,1, MPI_INT,MPI_COMM_WORLD);
   //CALI_MARK_END(comm_small);
   //CALI_MARK_BEGIN(comp_small);
   local_rdspls[0] = 0;
   for(i = 0 ; i< p-1 ;i++)
   {
      local_rdspls[i+1] = local_rdspls[i] + local_rscounts[i];
   }
   //CALI_MARK_END(comp_small);
   
   
   //CALI_MARK_BEGIN(comm_large);
   MPI_Alltoallv(local_bucketlist, local_sscounts, local_sdspls, MPI_DOUBLE, local_array,local_rscounts, local_rdspls, MPI_DOUBLE, MPI_COMM_WORLD);
   //CALI_MARK_END(comm_large);
   

   //CALI_MARK_BEGIN(comp_large);
   qsort_dbls(local_array, local_array_sizes[rank]);
   //CALI_MARK_END(comp_large);
   sorted_array = new double[n];


   fdspls[0] = 0;
   for(i = 0 ; i< p-1 ;i++)
   {
      fdspls[i+1] = fdspls[i] + local_array_sizes[i];
   }
   //CALI_MARK_BEGIN(comm_large);
   MPI_Gatherv(local_array, local_array_sizes[rank], MPI_DOUBLE, 
               sorted_array, local_array_sizes, fdspls, MPI_DOUBLE,0, MPI_COMM_WORLD); 

   //CALI_MARK_END(comm_large);
   if (rank == 0)
   {
      //CALI_MARK_BEGIN(correctness_check);
      if (verify(sorted_array, n-n%p))
         printf("Successful sort\n");
      else
        printf("Unsuccessful sort\n");
      //CALI_MARK_END(correctness_check);

   
   }


  
   MPI_Finalize();
   return 0;
   
}
