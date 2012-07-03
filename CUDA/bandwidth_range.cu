#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<math.h>

#define START atoi(argv[1])
#define END atoi(argv[2])
#define STEP atoi(argv[3])

void safe_call(cudaError_t ret, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		exit(-1);
	}
}

void fill_mat(char *arr, int size)
{
	int i;
	for(i=0;i<size;i++)
		arr[i] = 'a';
}

int main(int argc, char **argv)
{
	if(argc!=4)
	{
		printf("Syntax : exec <starting size> <end size> <step size>\n");
		exit(-1);
	}

	int i,j;
	int SIZE = ceil((END-START)/(STEP*sizeof(char)))+1;
		
	double *h2h, *h2d, *d2d, *d2h;
	h2h = (double *) malloc(SIZE*sizeof(double));
	h2d = (double *) malloc(SIZE*sizeof(double));
	d2d = (double *) malloc(SIZE*sizeof(double));
	d2h = (double *) malloc(SIZE*sizeof(double));
	
	char *h_A, *h_B;
	char *d_A, *d_B;
	
	cudaEvent_t start, stop;

	double time, bandwidth;	
	float diff;

	double time_start, time_end;
        struct timeval tv;
        struct timezone tz;
	
	safe_call(cudaEventCreate(&start),__LINE__);
	safe_call(cudaEventCreate(&stop),__LINE__);

	for(i=START,j=0;i<=END;i+=STEP)
	{	
		h_A = (char *) malloc(i);
		h_B = (char *) malloc(i);

		if(h_A==NULL || h_B==NULL)
		{
			printf("Error : host memory allocation\n");
			exit(-1);
		}

		safe_call(cudaMalloc((void **)&d_A, i),__LINE__);
		safe_call(cudaMalloc((void **)&d_B, i),__LINE__);

		fill_mat(h_A,i);	

		/*gettimeofday(&tv, &tz);
		time_start = (char)tv.tv_sec + (char)tv.tv_usec / 1000000.0;
		memcpy((void *)h_B, (void *)h_A, i); 
		gettimeofday(&tv, &tz);
		time_end = (char)tv.tv_sec + (char)tv.tv_usec / 1000000.0;
		bandwidth = ( i * sizeof(char) * 2.0 ) / ( 1024 * 1024 * ( time_end - time_start ) ) ;	
		h2h[j] = bandwidth;
		*/
		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpy((void *)d_A, (void *)h_A, i, cudaMemcpyHostToDevice),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( i * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
		h2d[j] = bandwidth;

		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpy((void *)d_B, (void *)d_A, i, cudaMemcpyDeviceToDevice),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( i * 2.0 ) / ( 1024 * 1024 * time ) ;	
		d2d[j] = bandwidth;

		safe_call(cudaEventRecord(start, 0),__LINE__);
		safe_call(cudaMemcpy((void *)h_B, (void *)d_B, i, cudaMemcpyDeviceToHost),__LINE__);
		safe_call(cudaEventRecord(stop, 0),__LINE__);
		safe_call(cudaEventSynchronize(stop),__LINE__);
		safe_call(cudaEventElapsedTime(&diff,start,stop),__LINE__);
		time = diff*1.0e-3;	
		bandwidth = ( i * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
		d2h[j] = bandwidth;

		safe_call(cudaFree(d_A),__LINE__);
		safe_call(cudaFree(d_B),__LINE__);

		free(h_A);
		free(h_B);

		j++;
	}

/*	printf("CPU Memcpy H2H\n");
	for(i=START,j=0;i<=END;i+=STEP)
	{
		printf("Size = %dB Bandwidth = %f MB/s\n",i,h2h[j++]);
	}
*/
	printf("cuda Memcpy H2D\n");
	for(i=START,j=0;i<=END;i+=STEP)
	{
		printf("Size = %dB Bandwidth = %f MB/s\n",i,h2d[j++]);
	}

	printf("cuda Memcpy D2D\n");
	for(i=START,j=0;i<=END;i+=STEP)
	{
		printf("Size = %dB Bandwidth = %f MB/s\n",i,d2d[j++]);
	}

	printf("cuda Memcpy D2H\n");
	for(i=START,j=0;i<=END;i+=STEP)
	{
		printf("Size = %dB Bandwidth = %f MB/s\n",i,d2h[j++]);
	}
	
	safe_call(cudaEventDestroy(start),__LINE__);	
	safe_call(cudaEventDestroy(stop),__LINE__);
	
	free(h2h);
	free(h2d);
	free(d2d);
	free(d2h);
		
	return 0;
}
