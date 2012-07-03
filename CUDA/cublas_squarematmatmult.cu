// Author : Kapil Agarwal
// Date : 13 June 2012
// Perform matrix-matrix multiplication using CUBLAS library and compare the performance with CPU
// To compile : nvcc -lcublas cublas_matmatmult.cu -o cublas_matmatmult
// To exectute : ./cublas_matmatmult <square matrix dimension>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<cuda.h>
#include "cublas_v2.h"

#define ERROR 1.0e-9

// check cuda error after every cuda instruction
void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

// check cublas status after every cublas instruction
void safe_call_blas(cublasStatus_t ret, int line)
{
	if(ret!=CUBLAS_STATUS_SUCCESS)
	{
		printf("Error at line %d : %s\n",line);
		exit(-1);
	}
}

// fill matrix with random values (column major order)
void fill_mat(double *arr, int len)
{
	int i;
	for(i=0;i<len;i++)
		arr[i] = drand48();
}

// display matrix (row major order)
void display(double *arr, int len)
{
	int i,j;
	for(i=0;i<len;i++)
	{
		for(j=0;j<len;j++)
			printf("%f\t",arr[j*len+i]);
		printf("\n");
	}
}

// column major computation on CPU because cublas does it in column major format
void cpu_matmatmul(double *a, double *b, double *c, int len)
{
	int i,j,k;
	double result;
	for(i=0;i<len;i++)
		for(j=0;j<len;j++)
		{
			result = 0.0;
			for(k=0;k<len;k++)
				result += (a[k*len+i] * b[j*len+k]);
			c[j*len+i] = result;
		}
}

// check for error between CPU and CUBLAS computation
void check(double *a, double *b, int len)
{
	int i;
	for(i=0;i<len*len;i++)
		if(fabs(a[i]-b[i])>ERROR)
		{
			printf("Error : CPU and GPU result do not match index=%d\tdevice=%f\thost=%f\n",i,a[i],b[i]);
			exit(-1);
		}
}

int main(int argc, char *argv[])
{
    // initial declarations
	cublasHandle_t handle;
	double *h_A, *h_B, *h_C, *cpu_C;
	double *d_A, *d_B, *d_C;
	int matdim, matlen;
	cudaEvent_t start,stop;
	float diff;
	double time,gflops,speedup,alpha=1,beta=0;	
	double time_start, time_end;
    struct timeval tv;
    struct timezone tz;
    
    // create cublas handle
	safe_call_blas(cublasCreate(&handle),__LINE__);
	
    // check syntax for execution
	if(argc!=2)
	{
		printf("Syntax : exec <sqr mat dim>\n");
		exit(-1);
	}
	
	matdim = atoi(argv[1]); // square matrix dimension
	matlen = matdim * matdim; // size after conversion of matrix to 1-D vector

    // create CUDA events for time measurement
	safe_call(cudaEventCreate(&start),__LINE__);
	safe_call(cudaEventCreate(&stop),__LINE__);

    // allocate host memory
	h_A = (double *) malloc(matlen*sizeof(double));
	h_B = (double *) malloc(matlen*sizeof(double));
	h_C = (double *) malloc(matlen*sizeof(double));

	if(h_A==NULL || h_B==NULL || h_C==NULL)
	{
		printf("Error : host memory allocation\n");
		exit(-1);
	}

    // allocate device memory
	safe_call(cudaMalloc((void **)&d_A, matlen*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_B, matlen*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_C, matlen*sizeof(double)),__LINE__);
	
    // fill matrix with random values
 	fill_mat(h_A,matlen);	
 	fill_mat(h_B,matlen);		

    // set cublas matrix
	safe_call_blas(cublasSetMatrix(matdim,matdim,sizeof(*h_A),h_A,matdim,d_A,matdim),__LINE__);
	safe_call_blas(cublasSetMatrix(matdim,matdim,sizeof(*h_B),h_B,matdim,d_B,matdim),__LINE__);

    // start time
	safe_call(cudaEventRecord(start, 0),__LINE__);
	
    // call cublas dgemm function for matrix multiplication
	safe_call_blas(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,matdim,matdim,matdim,&alpha,d_A,matdim,d_B,matdim,&beta,d_C,matdim),__LINE__);		
    // cuda synchronizations
	safe_call(cudaThreadSynchronize(),__LINE__);	
    // stop time
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	
    // calulate time taken
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	

    // get product matrix from device
	safe_call_blas(cublasGetMatrix(matdim,matdim,sizeof(*h_C),d_C,matdim,h_C,matdim),__LINE__);
	
    // perform multiplication on CPU and measure time taken
	if(matdim<=512)
	{
		cpu_C = (double *) malloc(matlen*sizeof(double));
		gettimeofday(&tv, &tz);
		time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
		cpu_matmatmul(h_A,h_B,cpu_C,matdim);
		gettimeofday(&tv, &tz);
		time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

		// check for errors between CPU and GPU results
		check(h_C,cpu_C,matdim);

		// measure speedup
		speedup = (time_end - time_start)/time;	
		/*
		// display matrices
		printf("\nMatrix A\n"); display(h_A,matdim);
		printf("\nMatrix B\n"); display(h_B,matdim);
		printf("\nHost Matrix C\n"); display(cpu_C,matdim);
		printf("\nGPU Matrix C\n"); display(h_C,matdim);
		 */    
		free(cpu_C);	
	}

    // free host memory
	free(h_A);
	free(h_B);
	free(h_C);

    // free device memory
	safe_call(cudaFree(d_A),__LINE__);
	safe_call(cudaFree(d_B),__LINE__);
	safe_call(cudaFree(d_C),__LINE__);

    // measure gflops
	gflops=(1.0e-9 * (( 2.0 * matdim * matdim * matdim )/time));
	printf("Success\nGPU Time = %lfs\nGflops = %f\n",time,gflops);
	if(matdim<=512)
	{
		printf("CPU Time = %lfs\nSpeedup = %lfx\n",time_end - time_start,speedup);	
	}

    // destroy cublas handle
	safe_call_blas(cublasDestroy(handle),__LINE__);

	return 0;
}

