/*****************************************************************************

                         C-DAC Tech Workshop : HEMPA-2011
                             Oct 17 - 21, 2011

  Example     : CUBlasSMatMatMult.cu
 
  Objective   : Write a CUDA Program for Matrix Matrix multiplication 
                using CUBLAS3 library function calls. 

  Input       : None 

  Output      : Execution time in seconds , Gflops achieved
                                                                                                                            
  Created     : Aug 2011    

  E-mail      : betatest@cdac.in         
                                 
****************************************************************************/


#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<sys/time.h>
#include "cublas.h"

#define SIZE 1024
#define EPS 1.0e-15

cudaEvent_t start,stop;
cudaError_t ret;
cublasStatus status;
cudaDeviceProp deviceProp;

double  *host_MatA,*host_MatB,*host_Res,*cpu_Res;
double  *device_MatA,*device_MatB,*device_Res;
int     RowA,ColA,ColB;
float  Tsec;
float   elapsedTime;
int size=SIZE;	

/* checking GPU all kind of ERROR*/
#define CUBLAS_SAFE_CALL(call)					\
	status=call;						\
	if(status != CUBLAS_STATUS_SUCCESS)			\
	 { printf(" Error in CUBLAS call.Program terminating\n");\
	    exit(-1);						\
	 }					



/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);                                        
        switch(ret)
        {
                case cudaSuccess:
                //              printf("Success\n");                    
                                break;
        /*      case cudaErrorInvalidValue:                             
                                {
                                printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                exit(-1);
                                break;  
                                }                       
                case cudaErrorInvalidDevicePointer:                     
                                {
                                printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }                       
                case cudaErrorInvalidMemcpyDirection:                   
                                {
                                printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);        
                                exit(-1);
                                break;
                                }                       */
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}



/*Get the number of GPU devices present on the host */
int get_DeviceCount()
{
        int count;
        cudaGetDeviceCount(&count);
        return count;
}


/*Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}


/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        exit(-1);
}


/*prints the result in screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
        else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}

/*function to calculate relative error*/
void relError(double* dRes,double* hRes,int size)
{
        double relativeError=0.0,errorNorm=0.0;
        int flag=0;
        int i;

        for( i = 0; i < size; ++i) {
                if (fabs(hRes[i]) > fabs(dRes[i]))
                        relativeError = fabs((hRes[i] - dRes[i]) / hRes[i]);
                else
                        relativeError = fabs((dRes[i] - hRes[i]) / dRes[i]);

                if (relativeError > EPS && relativeError != 0.0e+00 )
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
                        }
                }

        }
        if( flag == 1)
        {
                printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", EPS);
                printf(" \n Relative Error                  : %e\n", errorNorm);

        }
        else
                printf("\n Results verfication : Success\n");

}



/* sequential mat mat multiplication */
void CPU_MatMat()
{
	cpu_Res = (double *)malloc(RowA*ColB*sizeof(double));
	 if(cpu_Res==NULL)
                mem_error("host_Res","matmatmul",RowA*ColB,"double");


	int i,j;
	for(i=0;i<RowA;i++)
	for(j=0;j<ColB;j++)
	{
	int k;
	cpu_Res[i+ColB*j]=0.00;
	for(k=0;k<ColA;k++)
	cpu_Res[i+ColB*j]+=host_MatA[i+RowA*k]*host_MatB[j*ColA+k];
	}
}

/*calculate Gflops*/
double calculate_gflops(float &Tsec)
{
        float gflops=(1.0e-9 * (( 2.0 *size*size*size )/Tsec));
        return gflops;
}



/*********************************************************************

          CUBLAS MAT-MAT MULTIPLICATION

**********************************************************************/

/* launch kernel*/
void launch_Cublas_dp_MatMat()
{
	double alpha=1.0;
	double beta=0.0;
	int lda,ldb,ldc;
        lda=RowA;
        ldb=ColA;
        ldc=ColB;
	cublasDgemm ('N', 'N', RowA , ColB ,ColA  , alpha, device_MatA, lda, device_MatB, ldb , beta,device_Res, ldc );
}

/* main function*/
int main(int argc, char **argv)
{
 	int device_Count=get_DeviceCount();
        printf("\n\nNUmber of Devices : %d\n\n", device_Count);

        // Device Selection, Device 1: Tesla C1060
        cudaSetDevice(0);

        int device;
        // Current Device Detection
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);

	// Vector length , Matrix Row and Col sizes..............
      	RowA=ColA=ColB=size;
    
	//printf("this programs does computation of square matrix only\n");

 	/*allocating the memory for each matrix */
        host_MatA =(double *)malloc(RowA*ColA*sizeof(double));
        host_MatB = (double *)malloc(ColA*ColB*sizeof(double));
        host_Res = (double *)malloc(RowA*ColB*sizeof(double));

	
	// ---------------checking host memory  for error..............................
	 if(host_MatA==NULL)
                mem_error("host_MatA","matmatmul",RowA*ColA,"double");

         if(host_MatB==NULL)
                mem_error("host_MatB","matmatmul",ColA*ColB,"double");

         if(host_Res==NULL)
                mem_error("host_Res","matmatmul",RowA*ColB,"double");
	
	//--------------Initializing the input arrays..............
        fill_dp_vector(host_MatA,RowA*ColA);
        fill_dp_vector(host_MatB,ColA*ColB);


  	/* allocate memory for GPU events */
	start = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
	stop = (cudaEvent_t) malloc (sizeof(cudaEvent_t));	
	if(start==NULL)
                mem_error("start","matmatmul",1,"cudaEvent_t");
        if(stop==NULL)
                mem_error("stop","matmatmul",1,"cudaEvent_t");
	
  	//event creation...
	CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));

  	//allocating memory on GPU
	CUBLAS_SAFE_CALL(cublasAlloc (RowA*ColA, sizeof(double), (void**)&device_MatA));
	CUBLAS_SAFE_CALL(cublasAlloc (ColA*ColB, sizeof(double), (void**)&device_MatB));
	CUBLAS_SAFE_CALL(cublasAlloc (RowA*ColB, sizeof(double), (void**)&device_Res));
 	
	// Initialization of vectors with host vectors 
	CUBLAS_SAFE_CALL(cublasSetVector (RowA*ColA, sizeof(double), host_MatA, 1, device_MatA, 1));
	CUBLAS_SAFE_CALL(cublasSetVector (ColA*ColB, sizeof(double), host_MatB, 1, device_MatB, 1));
	
	// Launching CUBLAS call.........
	CUBLAS_SAFE_CALL(cublasGetError());
	
	CUDA_SAFE_CALL(cudaEventRecord (start, 0)); 
	launch_Cublas_dp_MatMat();//...........cublas_dgemm is called....

	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize (stop));
	CUBLAS_SAFE_CALL(cublasGetError());
	
	//retriving result from device
       	CUBLAS_SAFE_CALL(cublasGetVector (RowA*ColB, sizeof(double), device_Res, 1, host_Res, 1));

	//computing elapsed time	
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	Tsec = elapsedTime *1.0e-3; //time in sec now

	// calling funtion for measuring Gflops

        calculate_gflops(Tsec);

	//printing the result on screen
    	print_on_screen("CUBLAS MAT MAT MULTIPLICATION",Tsec,calculate_gflops(Tsec),size,1);

	// CPU calculation..and checking error deviation....
        CPU_MatMat();
  	relError(cpu_Res,host_Res,size*size);

	/*free the memory of CUBLAS */
	CUBLAS_SAFE_CALL(cublasFree(device_MatA));
	CUBLAS_SAFE_CALL(cublasFree(device_MatB));
	CUBLAS_SAFE_CALL(cublasFree(device_Res));
	// ending CUBLAS routines...
	CUBLAS_SAFE_CALL(cublasShutdown());
	/* Free the Host memory */
	free(host_MatA);
	free(host_MatB);
	free(host_Res);
	free(cpu_Res);
	return 0;
}// end of main
