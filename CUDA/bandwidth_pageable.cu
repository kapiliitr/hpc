#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>

#define SIZE atoi(argv[1])

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

int main(int argc, char **argv)
{
	if(argc!=2)
	{
		printf("Syntax : exec <size>\n");
		exit(-1);
	}
	
	double *h_A, *h_B;
	double *d_A, *d_B;
	
	cudaEvent_t start, stop;

	double time, bandwidth;	
	float diff;

	double time_start, time_end;
        struct timeval tv;
        struct timezone tz;
	
	safe_call(cudaEventCreate(&start),__LINE__);
	safe_call(cudaEventCreate(&stop),__LINE__);

	h_A = (double *) malloc(SIZE*sizeof(double));
	h_B = (double *) malloc(SIZE*sizeof(double));

	if(h_A==NULL || h_B==NULL)
	{
		printf("Error : host memory allocation\n");
		exit(-1);
	}

	safe_call(cudaMalloc((void **)&d_A, SIZE*sizeof(double)),__LINE__);
	safe_call(cudaMalloc((void **)&d_B, SIZE*sizeof(double)),__LINE__);

	fill_mat(h_A,SIZE);	

	gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
        memcpy((void *)h_B, (void *)h_A, SIZE*sizeof(double)); 
	gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * ( time_end - time_start ) ) ;	
	printf("CPU Memcpy H2H Bandwidth = %f GB/s\n",bandwidth);

	safe_call(cudaEventRecord(start, 0),__LINE__);
	safe_call(cudaMemcpy((void *)d_A, (void *)h_A, SIZE*sizeof(double), cudaMemcpyHostToDevice),__LINE__);
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	
	bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
	printf("CUDA Memcpy H2D Bandwidth = %f GB/s\n",bandwidth);

	safe_call(cudaEventRecord(start, 0),__LINE__);
	safe_call(cudaMemcpy((void *)d_B, (void *)d_A, SIZE*sizeof(double), cudaMemcpyDeviceToDevice),__LINE__);
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	
	bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
	printf("CUDA Memcpy D2D Bandwidth = %f GB/s\n",bandwidth);

	safe_call(cudaEventRecord(start, 0),__LINE__);
	safe_call(cudaMemcpy((void *)h_B, (void *)d_B, SIZE*sizeof(double), cudaMemcpyDeviceToHost),__LINE__);
	safe_call(cudaEventRecord(stop, 0),__LINE__);
	safe_call(cudaEventSynchronize(stop),__LINE__);
	safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
	time = diff*1.0e-3;	
	bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
	printf("CUDA Memcpy D2H Bandwidth = %f GB/s\n",bandwidth);

	safe_call(cudaEventDestroy(start),__LINE__);	
	safe_call(cudaEventDestroy(stop),__LINE__);

	safe_call(cudaFree(d_A),__LINE__);
	safe_call(cudaFree(d_B),__LINE__);
	
	free(h_A);
	free(h_B);
	
	return 0;
}
