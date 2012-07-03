// Author : Kapil Agarwal
// Date : 14 June 2012
// Perform matrix-matrix multiplication using CUBLAS library and compare the performance with CPU
// To compile : nvcc -lcublas cublas_matmatmult.cu -o cublas_matmatmult
// To exectute : ./cublas_matmatmult <square matrix dimension>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<cuda.h>
#include<cublas_v2.h>

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
void fill_mat(double *arr, int h, int w)
{
	int i;
	for(i=0;i<h*w;i++)
		arr[i] = drand48();
}

// display matrix (row major order)
void display(double *arr, int h, int w)
{
	int i,j;
	for(i=0;i<h;i++)
	{
		for(j=0;j<w;j++)
			printf("%f\t",arr[i*w+j]);
		printf("\n");
	}
}

// row major computation on CPU because cublas does it in column major format
void cpu_matmatmul(double *a, double *b, double *c, int len_a, int len_b, int len_c)
{
	int i,j,k;
	double result;
	for(i=0;i<len_a;i++)
		for(j=0;j<len_c;j++)
		{
			result = 0.0;
			for(k=0;k<len_b;k++)
			{
				result += (a[i*len_b+k] * b[k*len_c+j]);
//				printf("multiplied %f and %f\n",a[i*len_b+k],b[k*len_c+j]);
			}
			c[i*len_c+j] = result;
		}
}

// check for error between CPU and CUBLAS computation
void check(double *a, double *b, int h, int w)
{
	int i;
	for(i=0;i<h*w;i++)
		if(fabs(a[i]-b[i])>ERROR)
		{
			printf("Error : CPU and GPU result do not match index=%d\tdevice=%f\thost=%f\n",i,a[i],b[i]);
			exit(-1);
		}
}

void transpose(double *a, int h, int w)
{
	int i,j;
	double *b;

	b = (double *) malloc(h*w*sizeof(double));

	for(i=0;i<h;i++)
		for(j=0;j<w;j++)
			b[j*h+i] = a[i*w+j];	
	for(i=0;i<h*w;i++)
		a[i] = b[i];
	
	free(b);
}

int main(int argc, char *argv[])
{
    // initial declarations
	cublasHandle_t handle;
	double *h_A, *h_B, *h_C, *cpu_C;
	double *d_A, *d_B, *d_C;
	int p, q, r;
	cudaEvent_t start,stop;
	float diff;
	double time,gflops,speedup,alpha=1,beta=0;	
	double time_start, time_end;
    struct timeval tv;
    struct timezone tz;
    
    // create cublas handle
	safe_call_blas(cublasCreate(&handle),__LINE__);
	
    // check syntax for execution
	if(argc!=4)
	{
		printf("Syntax : exec <p> <q> <r>\n");
		exit(-1);
	}
	
	p = atoi(argv[1]);	
	q = atoi(argv[2]);	
	r = atoi(argv[3]);	

    // create CUDA events for time measurement
	safe_call(cudaEventCreate(&start),__LINE__);
	safe_call(cudaEventCreate(&stop),__LINE__);

    // allocate host memory
	h_A = (double *) malloc(p*q*sizeof(double));
	h_B = (double *) malloc(q*r*sizeof(double));
	h_C = (double *) malloc(p*r*sizeof(double));

	if(h_A==NULL || h_B==NULL || h_C==NULL)
	{
		printf("Error : host memory allocation\n");
		exit(-1);
	}

    // allocate device memory
	safe_call(cudaMalloc((void **)&d_A, p*q*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_B, q*r*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_C, p*r*sizeof(double)),__LINE__);
	
    // fill matrix with random values
 	fill_mat(h_A,p,q);	
 	fill_mat(h_B,q,r);		

    // Display Initial Matrices
	printf("\nMatrix A\n"); display(h_A,p,q);
	printf("\nMatrix B\n"); display(h_B,q,r);

    // perform multiplication on CPU and measure time taken
	cpu_C = (double *) malloc(p*r*sizeof(double));
	gettimeofday(&tv, &tz);
	time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	cpu_matmatmul(h_A,h_B,cpu_C,p,q,r);
	gettimeofday(&tv, &tz);
	time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    // Display Host product
	printf("\nHost Matrix C\n"); display(cpu_C,p,r);

    // convert the matrix to column major order
	transpose(h_A,p,q);
	transpose(h_B,q,r);

    // set cublas matrix
	safe_call_blas(cublasSetMatrix(q,p,sizeof(*h_A),h_A,q,d_A,q),__LINE__);
	safe_call_blas(cublasSetMatrix(r,p,sizeof(*h_B),h_B,r,d_B,r),__LINE__);

/*
   // set cublas matrix
	safe_call_blas(cublasSetMatrix(p,q,sizeof(*h_A),h_A,p,d_A,p),__LINE__);
	safe_call_blas(cublasSetMatrix(q,r,sizeof(*h_B),h_B,q,d_B,q),__LINE__);
*/

    // start time
	safe_call(cudaEventRecord(start, 0),__LINE__);
	
    // call cublas dgemm function for matrix multiplication
    	safe_call_blas(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,r,q,&alpha,d_A,p,d_B,q,&beta,d_C,p),__LINE__);		

    // cuda synchronizations
	safe_call(cudaThreadSynchronize(),__LINE__);	
    // stop time
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	
    // calulate time taken
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	

    // get product matrix from device
	safe_call_blas(cublasGetMatrix(p,r,sizeof(*h_C),d_C,p,h_C,p),__LINE__);

    // convert product matrix back to row major
	//transpose(h_C,r,p);

    // display GPU product
	printf("\nGPU Matrix C\n"); display(h_C,p,r);
    
    // check for errors between CPU and GPU results
        check(h_C,cpu_C,p,r);

    // measure speedup
    speedup = (time_end - time_start)/time;	
    
    // free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(cpu_C);	

    // free device memory
	safe_call(cudaFree(d_A),__LINE__);
	safe_call(cudaFree(d_B),__LINE__);
	safe_call(cudaFree(d_C),__LINE__);

    // measure gflops
	gflops=(1.0e-9 * (( 1.0 * p * q * r )/time));
	printf("Success\nGPU Time = %lfs\nGflops = %f\nCPU Time = %lfs\nSpeedup = %lfx\n",time,gflops,time_end - time_start,speedup);	

    // destroy cublas handle
	safe_call_blas(cublasDestroy(handle),__LINE__);

	return 0;
}
