#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<string.h>
#include<assert.h>

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
	int SIZE, MODE, i; // 0=pageable 1=pinned
	char memmode[10], tempmode[10]; 	

	if(argc<2 || argc>3)
	{
		printf("Syntax : exec -<memory mode> <size>\n");
		exit(-1);
	}
	else if(argc==2)
	{
		MODE = 0;	
		SIZE = atoi(argv[1]);
	}
	else if(argc==3)
	{
		strcpy(tempmode,argv[1]);
		i=0;
		while(tempmode[i]=='-') { i++; }
		if(i==0)
		{
			printf("Syntax : exec -<memory mode> <size>\n");
			exit(-1);
		}
		strcpy(memmode,&tempmode[i]);
		if(strcmp(memmode,"pinned") == 0)
			MODE = 1;
		else if(strcmp(memmode,"pageable") == 0)
			MODE = 0;
		else
		{
			printf("Memory modes pinned and pageable only\n");
			exit(-1);
		}
		SIZE = atoi(argv[2]);
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
	
	if(MODE==0) //if memory mode = pageable
	{
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
		
		printf("Pageable Memory\n");		

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
		
		for(i=0;i<SIZE;i++)
			assert(h_A[i]==h_B[i]);

		safe_call(cudaFree(d_A),__LINE__);
		safe_call(cudaFree(d_B),__LINE__);

		free(h_A);
		free(h_B);
	}
	else //if memory mode = pinned
	{
		safe_call(cudaMallocHost((void **)&h_A, SIZE*sizeof(double)),__LINE__);
		safe_call(cudaMallocHost((void **)&h_B, SIZE*sizeof(double)),__LINE__);
	
		safe_call(cudaMalloc((void **)&d_A, SIZE*sizeof(double)),__LINE__);
		safe_call(cudaMalloc((void **)&d_B, SIZE*sizeof(double)),__LINE__);

		fill_mat(h_A,SIZE);	

		printf("Pinned Memory\n");		

		gettimeofday(&tv, &tz);
		time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
		memcpy((void *)h_B, (void *)h_A, SIZE*sizeof(double)); 
		gettimeofday(&tv, &tz);
		time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
		bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * ( time_end - time_start ) ) ;	
		printf("CPU Memcpy H2H Bandwidth = %f GB/s\n",bandwidth);

		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpyAsync((void *)d_A, (void *)h_A, SIZE*sizeof(double), cudaMemcpyHostToDevice, 0),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
		printf("CUDA Memcpy H2D Bandwidth = %f GB/s\n",bandwidth);

		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpyAsync((void *)d_B, (void *)d_A, SIZE*sizeof(double), cudaMemcpyDeviceToDevice, 0),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
		printf("CUDA Memcpy D2D Bandwidth = %f GB/s\n",bandwidth);

		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpyAsync((void *)h_B, (void *)d_B, SIZE*sizeof(double), cudaMemcpyDeviceToHost, 0),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( SIZE * sizeof(double) * 2.0 ) / ( 1024 * 1024 * 1024 * time ) ;	
		printf("CUDA Memcpy D2H Bandwidth = %f GB/s\n",bandwidth);

		for(i=0;i<SIZE;i++)
			assert(h_A[i]==h_B[i]);

		safe_call(cudaFree(d_A),__LINE__);
		safe_call(cudaFree(d_B),__LINE__);
	
		safe_call(cudaFreeHost(h_A),__LINE__);
		safe_call(cudaFreeHost(h_B),__LINE__);
	}

	safe_call(cudaEventDestroy(start),__LINE__);	
	safe_call(cudaEventDestroy(stop),__LINE__);

	return 0;
}
