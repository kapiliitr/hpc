#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>

void safe_call(cudaError_t ret)
{
	if(ret!=cudaSuccess)
	{
		printf("Error : %s\n",cudaGetErrorString(ret));
		exit(-1);
	}
}

void fill_vec(double *arr, int len)
{
	int i;
	for(i=0;i<len;i++)
		arr[i] = drand48();
}
__global__ void vecvecadd(double *a, double *b, double *c, int len)
{	
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
		c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
	double *h_A, *h_B, *h_C;
	double *d_A, *d_B, *d_C;
	int veclen, i, blockSize, gridSize;
	cudaEvent_t start,stop;
	float diff;
	double time,gflops;	
	
	if(argc!=3)
	{
		printf("Syntax : exec <veclen> <blocksize>\n");
		exit(-1);
	}
	
	veclen = atoi(argv[1]);
	blockSize = atoi(argv[2]);	
	gridSize = veclen/blockSize;
	if(veclen%blockSize==0)
		gridSize += 1;
	
	safe_call(cudaEventCreate(&start));
	safe_call(cudaEventCreate(&stop));

	h_A = (double *) malloc(veclen*sizeof(double));
	h_B = (double *) malloc(veclen*sizeof(double));
	h_C = (double *) malloc(veclen*sizeof(double));
	
	if(h_A==NULL || h_B==NULL || h_C==NULL)
	{
		printf("Error : host memory allocation\n");
		exit(-1);
	}

	safe_call(cudaMalloc((void **)&d_A, veclen*sizeof(double)));
	safe_call(cudaMalloc((void **)&d_B, veclen*sizeof(double)));
	safe_call(cudaMalloc((void **)&d_C, veclen*sizeof(double)));
	
 	fill_vec(h_A,veclen);	
 	fill_vec(h_B,veclen);	

	safe_call(cudaMemcpy((void *)d_A, (void *)h_A, veclen*sizeof(double), cudaMemcpyHostToDevice));
	safe_call(cudaMemcpy((void *)d_B, (void *)h_B, veclen*sizeof(double), cudaMemcpyHostToDevice));
	
	safe_call(cudaEventRecord(start, 0));
	vecvecadd<<<gridSize,blockSize>>>(d_A,d_B,d_C,veclen);
	safe_call(cudaEventRecord(stop, 0));
	safe_call(cudaEventSynchronize(stop));
	
	safe_call(cudaEventElapsedTime(&diff,start,stop));
	time = diff*1.0e-3;	

	safe_call(cudaMemcpy((void *)h_C, (void *)d_C, veclen*sizeof(double), cudaMemcpyDeviceToHost));
	
	for(i=0;i<veclen;i++)
		if(h_C[i]!=(h_A[i]+h_B[i]))
		{
			printf("Error in calculation\n");
			exit(-1);
		}

	safe_call(cudaFree(d_A));
	safe_call(cudaFree(d_B));
	safe_call(cudaFree(d_C));

	free(h_A);
	free(h_B);
	free(h_C);
	
	gflops=(1.0e-9 * (( 1.0 *veclen )/time));
	printf("Success\nTime = %lfs\nGflops = %f\n",time,gflops);	

	return 0;
}
