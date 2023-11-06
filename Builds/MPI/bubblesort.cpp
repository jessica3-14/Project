#include "helper.cpp"
#include <caliper/cali.h>
#include<caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

void bubbleSort(double* test, int dataSize) {
  bool swapped = true;
  while(swapped) {
    swapped = false;
    for(int i = 0; i < dataSize - 1; i++) {
      if(test[i] > test[i+1]) {
        std::swap(test[i], test[i+1]);
        swapped = true;
      }
    }
  }
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

const char* whole_computation = "whole_computation";
const char* master_initialization = "master_initialization";
const char* master_bubble = "master_bubble";
const char* worker_send = "worker_send";
const char* worker_receive = "worker_receive";
const char* worker_calculation = "worker_calculation";

double whole_comp_start_time, whole_comp_end_time, whole_comp_time;
double master_init_start_time, master_init_end_time, master_init_time;
double master_bubble_start_time, master_bubble_end_time, master_bubble_time;
double worker_send_start_time, worker_send_end_time, worker_send_time;
double worker_receive_start_time, worker_receive_end_time, worker_receive_time;
double worker_calc_start_time, worker_calc_end_time, worker_calc_time;

MPI_Comm workcom;
MPI_Comm_split(MPI_COMM_WORLD, taskid != 0, taskid, &workcom);

if (numprocs < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numprocs-1;

printf("Process %d is sending the array to workers\n", taskid);

//start whole computation
CALI_MARK_BEGIN(whole_computation);
whole_comp_start_time = MPI_Wtime();

cali::ConfigManager mgr;
mgr.start();

double* test = new double[1000];
int chunkSize = 1000 / (numprocs - 1);


if(taskid == MASTER){
  printf("bubble_sort has started with %d tasks.\n", numprocs);
  printf("****************************************************************\n");
  //start master initialization 
  CALI_MARK_BEGIN(master_initialization);
  master_init_start_time = MPI_Wtime();

  //genData(1000,1,test);
  for(int i = 0; i < 1000; i++){
    test[i] = rand() % 1000;
  }
  
  master_init_end_time = MPI_Wtime();
  master_init_time = master_init_end_time - master_init_start_time;
  CALI_MARK_END(master_initialization);
  //end master initialization
  
  //start master bubble
  CALI_MARK_BEGIN(master_bubble);
  master_bubble_start_time = MPI_Wtime();
  
  mtype = FROM_MASTER;
  std::cout << "sending array from master to worker" << std::endl;
  destination = 1;
  for(int dest = 1; dest < numprocs; dest++){
    MPI_Send(&chunkSize, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    MPI_Send(&test[(dest-1)*chunkSize], chunkSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
  }
  
  mtype = FROM_WORKER;
  std::cout << "receiving array from master to worker" << std::endl;
  for(int source = 1; source < numprocs; source++){
    MPI_Recv(&test[(source-1)*chunkSize], chunkSize, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
  }
  
  master_bubble_end_time = MPI_Wtime();
  master_bubble_time = master_bubble_end_time - master_bubble_start_time;
  CALI_MARK_END(master_bubble);
}
//end master bubble

if(taskid > MASTER){
  //start worker receive
  CALI_MARK_BEGIN(worker_receive);
  worker_receive_start_time = MPI_Wtime();
  
  mtype = FROM_MASTER;
  MPI_Recv(&chunkSize, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
  
  double* localData = new double[chunkSize];
  
  MPI_Recv(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
  
  worker_receive_end_time = MPI_Wtime();
  worker_receive_time = worker_receive_end_time - worker_receive_start_time;
  CALI_MARK_END(worker_receive);
  //end worker receive
  
  //start worker calculation
  CALI_MARK_BEGIN(worker_calculation);
  worker_calc_start_time = MPI_Wtime();
  
  bubbleSort(localData, chunkSize);
  
  worker_calc_end_time = MPI_Wtime();
  worker_calc_time = worker_calc_end_time - worker_calc_start_time;
  CALI_MARK_END(worker_calculation);
  //end worker calculation
  
  //start worker send
  CALI_MARK_BEGIN(worker_send);
  worker_send_start_time = MPI_Wtime();
  
  mtype = FROM_WORKER;
  MPI_Send(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  
  worker_send_end_time = MPI_Wtime();
  worker_send_time = worker_send_end_time - worker_send_start_time;
  CALI_MARK_END(worker_send);
  //end worker send

  delete[] localData;
}

//end whole computation
whole_comp_end_time = MPI_Wtime();
whole_comp_time = whole_comp_end_time - whole_comp_start_time;
CALI_MARK_END(whole_computation);

//int ret=0;
//check_sort(test,1000,&ret);
//printf("%d\n",ret);

adiak::init(NULL);
   adiak::user();
   adiak::launchdate();
   adiak::libraries();
   adiak::cmdline();
   adiak::clustername();
   adiak::value("num_procs", numprocs);


double worker_receive_time_max, worker_receive_time_min, worker_receive_time_sum, worker_receive_time_avg = 0;
double worker_send_time_max, worker_send_time_min, worker_send_time_sum, worker_send_time_avg = 0;
double worker_calc_time_max, worker_calc_time_min, worker_calc_time_sum, worker_calc_time_avg = 0;

MPI_Reduce(&worker_receive_time, &worker_receive_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
MPI_Reduce(&worker_receive_time, &worker_receive_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
MPI_Reduce(&worker_receive_time, &worker_receive_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);

MPI_Reduce(&worker_send_time, &worker_send_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
MPI_Reduce(&worker_send_time, &worker_send_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
MPI_Reduce(&worker_send_time, &worker_send_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);

MPI_Reduce(&worker_calc_time, &worker_calc_time_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
MPI_Reduce(&worker_calc_time, &worker_calc_time_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
MPI_Reduce(&worker_calc_time, &worker_calc_time_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);



if(taskid == 0){
  printf("\n");
  printf("Master times:\n");
  printf("whole computation time: %f\n", whole_comp_time);
  printf("master initialization time: %f\n", master_init_time);
  printf("master bubble time: %f\n", master_bubble_time);
  printf("\n");
  
  adiak::value("MPI_Reduce-whole-computation_time", whole_comp_time);
  adiak::value("MPI_Reduce-master_initialization_time", master_init_time);
  adiak::value("MPI_Reduce-master_bubble_time", master_bubble_time);
  
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
}
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

mgr.stop();
mgr.flush();

MPI_Finalize();
}
