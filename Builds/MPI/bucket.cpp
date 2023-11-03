#include "helper.cpp"



int main(int argc, char** argv){
MPI_Init(&argc,&argv);

double* test = new double[1000];
genData(1000,1,test);
int ret=0;
check_sort(test,1000,&ret);
printf("%d\n",ret);

MPI_Finalize();
}
