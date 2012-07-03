#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<math.h>

#define ERROR 1.0e-9

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

void fill_mat(double *arr, int len)
{
	int i;
	for(i=0;i<len;i++)
		arr[i] = drand48();
}

__global__ void gpu_matmatmul(double *a, double *b, double *c, int len)
{
	int i_x,i_y,j;	
	double prod;	
	
	i_x = blockDim.x*blockIdx.x+threadIdx.x;
	i_y = blockDim.y*blockIdx.y+threadIdx.y;
	
	prod = 0.0;	

	for(j=0;j<len;j++)
		prod += (a[i_x*len+j]*b[j*len+i_y]);
	
	c[i_x*len+i_y] = prod;

	__syncthreads();
}

void cpu_matmatmul(double *a, double *b, double *c, int len)
{
	int i,j,k;
	double result;
	for(i=0;i<len;i++)
		for(j=0;j<len;j++)
		{
			result = 0.0;
			for(k=0;k<len;k++)
				result += (a[i*len+k] * b[k*len+j]);
			c[i*len+j] = result;
		}
}

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

int main(int argc, char **argv)
{
	double *h_A, *h_B, *h_C, *cpu_C;
	double *d_A, *d_B, *d_C;
	int matdim, matlen, blockSize;
	cudaEvent_t start,stop;
	float diff;
	double time,gflops,speedup;	
	double time_start, time_end;
        struct timeval tv;
        struct timezone tz;
	
	if(argc!=3)
	{
		printf("Syntax : exec <sqr mat dim> <block size>\n");
		exit(-1);
	}
	
	matdim = atoi(argv[1]);
	blockSize = atoi(argv[2]);	
	
	if(blockSize>32)
	{
		printf("Maximum block size is 32\n");
		exit(-1);
	}
	
	if(matdim%blockSize!=0)
	{
		printf("matrix dimension should be multiple of block size\n");
		exit(-1);
	}
	matlen = matdim * matdim;

	dim3 bDim(blockSize,blockSize);
	dim3 gDim(matdim/blockSize,matdim/blockSize);
	
	safe_call(cudaEventCreate(&start),__LINE__);
	safe_call(cudaEventCreate(&stop),__LINE__);

	h_A = (double *) malloc(matlen*sizeof(double));
	h_B = (double *) malloc(matlen*sizeof(double));
	h_C = (double *) malloc(matlen*sizeof(double));

	if(h_A==NULL || h_B==NULL || h_C==NULL)
	{
		printf("Error : host memory allocation\n");
		exit(-1);
	}

	safe_call(cudaMalloc((void **)&d_A, matlen*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_B, matlen*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_C, matlen*sizeof(double)),__LINE__);
	
 	fill_mat(h_A,matlen);	
 	fill_mat(h_B,matlen);	

	safe_call(cudaMemcpy((void *)d_A, (void *)h_A, matlen*sizeof(double), cudaMemcpyHostToDevice),__LINE__);
	safe_call(cudaMemcpy((void *)d_B, (void *)h_B, matlen*sizeof(double), cudaMemcpyHostToDevice),__LINE__);
	
	safe_call(cudaEventRecord(start, 0),__LINE__);
	gpu_matmatmul<<<gDim,bDim>>>(d_A,d_B,d_C,matdim);
	safe_call(cudaThreadSynchronize(),__LINE__);	
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	

	safe_call(cudaMemcpy((void *)h_C, (void *)d_C, matlen*sizeof(double), cudaMemcpyDeviceToHost),__LINE__);
	
	cpu_C = (double *) malloc(matlen*sizeof(double));
        gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	cpu_matmatmul(h_A,h_B,cpu_C,matdim);
        gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	check(h_C,cpu_C,matdim);
	speedup = (time_end - time_start)/time;	

	safe_call(cudaEventDestroy(start),__LINE__);	
	safe_call(cudaEventDestroy(stop),__LINE__);

	safe_call(cudaFree(d_A),__LINE__);
	safe_call(cudaFree(d_B),__LINE__);
	safe_call(cudaFree(d_C),__LINE__);
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(cpu_C);	
	
	gflops=(1.0e-9 * (( 2.0 * matdim * matdim * matdim )/time));
	printf("Success\nGPU Time = %lfs\nGflops = %f\nCPU Time = %lfs\nSpeedup = %lfx\n",time,gflops,time_end - time_start,speedup);	

	return 0;
}
