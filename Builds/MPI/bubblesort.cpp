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

int verify(double *test, int dataSize){
  for(int i = 0; i < dataSize; i++){
    if(test[i] > test[i+1]){
      return -1;
    }
  }
  return 1;
}


int main(int argc, char** argv){
cali::ConfigManager mgr;
mgr.start();
CALI_MARK_BEGIN("main");

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
  MPI_Comm workcom;
  MPI_Comm_split(MPI_COMM_WORLD, taskid != 0, taskid, &workcom);
  
  double data_init_start, data_init_end, data_init_time = 0;
  double comm_start, comm_end, comm_time = 0;
  double comm_large_start, comm_large_end, comm_large_time = 0;
  double comm_small_start, comm_small_end, comm_small_time = 0;
  double MPI_Recv_start, MPI_Recv_end, MPI_Recv_time = 0;
  double comp_start, comp_end, comp_time = 0;
  double comp_large_start, comp_large_end, comp_large_time = 0;
  double MPI_Send_start, MPI_Send_end, MPI_Send_time = 0;
  double correctness_check_start, correctness_check_end, correctness_check_time = 0;
  
  if (numprocs < 2 ) {
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
    }
  numworkers = numprocs-1;
  
  double* test = new double[1000];
  int chunkSize = 1000 / (numprocs - 1);
  
  int dataSize = 1000;
  int mode = 1;
  if(argc > 1){
    mode = std::atoi(argv[1]);
  }
  if(argc > 2){
    dataSize = std::atoi(argv[2]);
  }
  test = new double[dataSize];
  chunkSize = dataSize / (numprocs - 1);
  
  
  if(taskid == MASTER){
    //start data initialization 
    CALI_MARK_BEGIN("data_init");
    data_init_start = MPI_Wtime();
    
    genData(dataSize, mode, test);
    
    data_init_end = MPI_Wtime();
    data_init_time = data_init_end - data_init_start;
    CALI_MARK_END("data_init");
    //end data initialization
    //MPI_Barrier(MPI_COMM_WORLD);
  
    CALI_MARK_BEGIN("comm");
    comm_start = MPI_Wtime();
    //start master bubble
    CALI_MARK_BEGIN("comm_large");
    comm_large_start = MPI_Wtime();
    
    mtype = FROM_MASTER;
    destination = 1;
    CALI_MARK_BEGIN("MPI_Send");
    MPI_Send_start = MPI_Wtime();
    
    for(int dest = 1; dest < numprocs; dest++){
      MPI_Send(&chunkSize, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&test[(dest-1)*chunkSize], chunkSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
    }
    
    MPI_Send_end = MPI_Wtime();
    MPI_Send_time = MPI_Send_end - MPI_Send_start;
    CALI_MARK_END("MPI_Send");
    
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN("MPI_Recv");
    MPI_Recv_start = MPI_Wtime();
    
    for(int source = 1; source < numprocs; source++){
      MPI_Recv(&test[(source-1)*chunkSize], chunkSize, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
    }
    
    MPI_Recv_end = MPI_Wtime();
    MPI_Recv_time = MPI_Recv_end - MPI_Recv_start;
    CALI_MARK_END("MPI_Recv");
  
    comm_large_end = MPI_Wtime();
    comm_large_time = comm_large_end - comm_large_start;
    CALI_MARK_END("comm_large");
    comm_end = MPI_Wtime();
    comm_time = comm_end - comm_start;
    CALI_MARK_END("comm");
  }
  //end master bubble
  
  if(taskid > MASTER){
    CALI_MARK_BEGIN("comm");
    comm_start = MPI_Wtime();
    //start worker receive
    CALI_MARK_BEGIN("comm_small");
    comm_small_start = MPI_Wtime();
    
    CALI_MARK_BEGIN("MPI_Recv");
    MPI_Recv_start = MPI_Wtime();
    
    mtype = FROM_MASTER;
    MPI_Recv(&chunkSize, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    double* localData = new double[chunkSize];
    MPI_Recv(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv_end = MPI_Wtime();
    MPI_Recv_time = MPI_Recv_end - MPI_Recv_start;
    CALI_MARK_END("MPI_Recv");
    
    comm_small_end = MPI_Wtime();
    comm_small_time = comm_small_end - comm_small_start;
    CALI_MARK_END("comm_small");
    //end worker receive
    
    comm_end = MPI_Wtime();
    comm_time = comm_end - comm_start;
    CALI_MARK_END("comm");
    
    //MPI_Barrier(MPI_COMM_WORLD);
    
    CALI_MARK_BEGIN("comp");
    comp_start = MPI_Wtime();
    //start worker calculation
    
    CALI_MARK_BEGIN("comp_large");
    comp_large_start = MPI_Wtime();
    
    bubbleSort(localData, chunkSize);
    
    comp_large_end = MPI_Wtime();
    comp_large_time = comp_large_end - comp_large_start;
    CALI_MARK_END("comp_large");
    //end worker calculation
    
    comp_end = MPI_Wtime();
    comp_time = comp_end - comp_start;
    CALI_MARK_END("comp");
    
    //MPI_Barrier(MPI_COMM_WORLD);
    
    
    //communication again
    CALI_MARK_BEGIN("comm");
    comm_start = MPI_Wtime();
    //start worker send
    CALI_MARK_BEGIN("comm_small");
    comm_small_start = MPI_Wtime();
    CALI_MARK_BEGIN("MPI_Send");
    MPI_Send_start = MPI_Wtime();
    
    mtype = FROM_WORKER;
    MPI_Send(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send_end = MPI_Wtime();
    MPI_Send_time = MPI_Send_end - MPI_Send_start;
    CALI_MARK_END("MPI_Send");
    comm_small_end = MPI_Wtime();
    comm_small_time = comm_small_end - comm_small_start;
    CALI_MARK_END("comm_small");
    //end worker send
    
    comm_end = MPI_Wtime();
    comm_time = comm_end - comm_start;
    CALI_MARK_END("comm");
    delete[] localData;
  }
  
  adiak::init(NULL);
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Bubblesort");
  adiak::value("ProgrammingModel", "MPI");
  adiak::value("Datatype", "double");
  adiak::value("SizeOfDatatype", 8);
  adiak::value("InputSize", dataSize);
  adiak::value("InputType", mode);
  adiak::value("num_procs", numprocs);
  adiak::value("group_num", 12);
  adiak::value("implementation_source", "https://www.geeksforgeeks.org/bubble-sort/");
  
  double data_init_min, data_init_max, data_init_avg, data_init_sum;
  double comm_min, comm_max, comm_avg, comm_sum;
  double comm_small_min, comm_small_max, comm_small_avg, comm_small_sum;
  double comm_large_min, comm_large_max, comm_large_avg, comm_large_sum;
  double MPI_Send_min, MPI_Send_max, MPI_Send_avg, MPI_Send_sum;
  double MPI_Recv_min, MPI_Recv_max, MPI_Recv_avg, MPI_Recv_sum;
  double comp_min, comp_max, comp_avg, comp_sum;
  double comp_large_min, comp_large_max, comp_large_avg, comp_large_sum;
  
  double comm_std, comm_small_std, MPI_Send_std, MPI_Recv_std, comp_std, comp_large_std;
  
  MPI_Reduce(&data_init_time, &data_init_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&data_init_time, &data_init_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&data_init_time, &data_init_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&comm_time, &comm_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&comm_time, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&comm_small_time, &comm_small_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&comm_small_time, &comm_small_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&comm_small_time, &comm_small_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&comm_large_time, &comm_large_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&comm_large_time, &comm_large_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&comm_large_time, &comm_large_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&MPI_Send_time, &MPI_Send_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&MPI_Send_time, &MPI_Send_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&MPI_Send_time, &MPI_Send_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&MPI_Recv_time, &MPI_Recv_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&MPI_Recv_time, &MPI_Recv_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&MPI_Recv_time, &MPI_Recv_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&comp_time, &comp_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&comp_time, &comp_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&comp_time, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  MPI_Reduce(&comp_large_time, &comp_large_min, 1, MPI_DOUBLE, MPI_MIN, MASTER, workcom);
  MPI_Reduce(&comp_large_time, &comp_large_max, 1, MPI_DOUBLE, MPI_MAX, MASTER, workcom);
  MPI_Reduce(&comp_large_time, &comp_large_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, workcom);
  
  if(taskid == 0){
  
    CALI_MARK_BEGIN("correctness_check");
    correctness_check_start = MPI_Wtime();
    if(verify(test, dataSize)){
      printf("Sort successful\n");
    }
    else{
      printf("Sort unsuccessful\n");
    }
    correctness_check_end = MPI_Wtime();
    correctness_check_time = correctness_check_end - correctness_check_start;
    CALI_MARK_END("correctness_check");
    
    
    printf("\n******************************************************\n");
    printf("Master Processes:\n");
    printf("Data Initialization time: %.6f\n", data_init_time);
    printf("Whole Communication time: %.6f\n", comm_time);
    printf("Large Communication time: %.6f\n", comm_large_time);
    printf("MPI Send time: %.6f\n", MPI_Send_time);
    printf("MPI Receive time: %.6f\n", MPI_Recv_time);
    printf("\n******************************************************\n");
    
    adiak::value("Data_initialization_time", data_init_time);
    adiak::value("Whole_communication_time", comm_time);
    adiak::value("Small_communcation_time", comm_small_time);
    adiak::value("Large_communication_time", comm_large_time);
    adiak::value("MPI_Send_time", MPI_Send_time);
    adiak::value("MPI_Receive_time", MPI_Recv_time);
    adiak::value("Correctness_check", correctness_check_time);
    
    mtype = FROM_WORKER;
    
    MPI_Recv(&comm_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&comm_small_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_small_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_small_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comm_small_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&MPI_Send_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Send_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Send_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Send_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&MPI_Recv_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Recv_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Recv_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&MPI_Recv_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&comp_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    MPI_Recv(&comp_large_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_large_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_large_avg, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&comp_large_std, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
    
    adiak::value("Whole_communication_time_min", comm_min);
    adiak::value("Whole_communication_time_max", comm_max);
    adiak::value("Whole_communication_time_avg", comm_avg);
    adiak::value("Whole_communication_time_std", comm_std);
    
    adiak::value("Small_communication_time_min", comm_small_min);
    adiak::value("Small_communication_time_max", comm_small_max);
    adiak::value("Small_communication_time_avg", comm_small_avg);
    adiak::value("Small_communication_time_std", comm_small_std);
    
    adiak::value("MPI_Send_time_min", MPI_Send_min);
    adiak::value("MPI_Send_time_max", MPI_Send_max);
    adiak::value("MPI_Send_time_avg", MPI_Send_avg);
    adiak::value("MPI_Send_time_std", MPI_Send_std);
    
    adiak::value("MPI_Recv_time_min", MPI_Recv_min);
    adiak::value("MPI_Recv_time_max", MPI_Recv_max);
    adiak::value("MPI_Recv_time_avg", MPI_Recv_avg);
    adiak::value("MPI_Recv_time_std", MPI_Recv_std);
    
    adiak::value("Whole_computation_time_min", comp_min);
    adiak::value("Whole_computation_time_max", comp_max);
    adiak::value("Whole_computation_time_avg", comp_avg);
    adiak::value("Whole_computation_time_std", comp_std);
    
    adiak::value("Large_computation_time_min", comp_large_min);
    adiak::value("Large_computation_time_max", comp_large_max);
    adiak::value("Large_computation_time_avg", comp_large_avg);
    adiak::value("Large_computation_time_std", comp_large_std);
  }
  else if(taskid == 1){
    comm_avg = comm_sum / (double)numworkers;
    comm_small_avg = comm_small_sum / (double)numworkers;
    MPI_Send_avg = MPI_Send_sum / (double)numworkers;
    MPI_Recv_avg = MPI_Recv_sum / (double)numworkers;
    comp_avg = comp_sum / (double)numworkers;
    comp_large_avg = comp_large_sum / (double)numworkers;
    
    comm_std = (comm_sum / numworkers - comm_avg * comm_avg) / numworkers;
    comm_small_std = (comm_small_sum / numworkers - comm_small_avg * comm_small_avg) / numworkers;
    MPI_Send_std = (MPI_Send_sum / numworkers - MPI_Send_avg * MPI_Send_avg) / numworkers;
    MPI_Recv_std = (MPI_Recv_sum / numworkers - MPI_Recv_avg * MPI_Recv_avg) / numworkers;
    comp_std = (comp_sum / numworkers - comp_avg * comp_avg) / numworkers;
    comp_large_std = (comp_large_sum / numworkers - comp_large_avg * comp_large_avg) / numworkers;
    
    
    printf("\n");
    printf("Worker Processes:\n");
    printf("Whole Communication time: %.6f\n", comm_time);
    printf("Whole Communication time min: %.6f \n", comm_min);
    printf("Whole Communication time max: %.6f \n", comm_max);
    printf("Whole Communication time average: %.6f \n", comm_avg);
    printf("Whole Communication time variance: %.6f\n", comm_std);
    
    printf("Small communication time: %.6f\n", comm_small_time);
    printf("Small communication time min: %.6f \n", comm_small_min);
    printf("Small communication time max: %.6f \n", comm_small_max);
    printf("Small communication time average: %.6f \n", comm_small_avg);
    printf("Small communication time variance: %.6f\n", comm_small_std);
    
    printf("MPI Send time: %.6f\n", MPI_Send_time);
    printf("MPI Send time min: %.6f \n", MPI_Send_min);
    printf("MPI Send time max: %.6f \n", MPI_Send_max);
    printf("MPI Send time average: %.6f \n", MPI_Send_avg);
    printf("MPI Send time variance: %.6f\n", MPI_Send_std);
    
    printf("MPI Receive time: %.6f\n", MPI_Recv_time);
    printf("MPI Recv time min: %.6f \n", MPI_Recv_min);
    printf("MPI Recv time max: %.6f \n", MPI_Recv_max);
    printf("MPI Recv time average: %.6f \n", MPI_Recv_avg);
    printf("MPI Recv time variance: %.6f\n", MPI_Recv_std);
    
    printf("Whole computation time: %.6f\n", comp_time);
    printf("Whole computation time min: %.6f \n", comp_min);
    printf("Whole computation time max: %.6f \n", comp_max);
    printf("Whole computation time average: %.6f \n", comp_avg);
    printf("Whole computation time variance: %.6f\n", comp_std);
    
    printf("Large computation time: %.6f\n", comp_large_time);
    printf("Large computation time min: %.6f \n", comp_large_min);
    printf("Large computation time max: %.6f \n", comp_large_max);
    printf("Large computation time average: %.6f \n", comp_large_avg);
    printf("Large computation time variance: %.6f\n", comp_large_std);
    
    mtype = FROM_WORKER;
    MPI_Send(&data_init_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&data_init_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&data_init_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&comm_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&comm_small_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_small_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_small_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_small_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&comm_large_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_large_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comm_large_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&MPI_Send_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Send_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Send_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Send_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&MPI_Recv_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Recv_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Recv_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&MPI_Recv_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&comp_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    
    MPI_Send(&comp_large_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_large_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_large_avg, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&comp_large_std, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  }
  
  CALI_MARK_END("main");
  mgr.stop();
  mgr.flush();
    
  MPI_Finalize();
}

