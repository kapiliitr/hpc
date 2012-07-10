#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<mpi.h>
#include<string.h>
#include<sys/time.h>

void safe_call(cudaError_t ret, int myrank, int line)
{
	if(ret!=cudaSuccess)
	{
		if(myrank == 0)
			printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		MPI_Finalize();
		exit(-1);
	}
}

void fill_data(char *arr, int len)
{
	int i;
	
	for(i=0;i<len;i++)
	{
		srand(time(NULL));
		arr[i] = (char)(rand()%26 + 97);
	}
}

int main(int argc, char *argv[])
{
	int 		comm_size, myrank, i;
	int 		START, END, STEP, SIZE;
	MPI_Status 	status;	
	char 		myname[MPI_MAX_PROCESSOR_NAME];
	int 		namelen, devcount, device;
	char		devname[256];
	cudaDeviceProp	devprop;
	char 		*h_A, *h_B;
	char 		*d_A, *d_B;
	cudaEvent_t 	start, stop;
	double 		time, h2d, d2d, d2h;	
	float 		diff;	

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	
	if(argc==4)
	{
		START = atoi(argv[1]);
		END = atoi(argv[2]);
		STEP = atoi(argv[3]);
	}
	else
	{
		START = 1024;
		END = 10240;
		STEP = 1024;
	}
	
	if(myrank == 0)
		printf("START=%d END=%d STEP=%d\n",START,END,STEP);

	MPI_Get_processor_name(myname, &namelen);
	myname[namelen++] = (char)0;         

	safe_call(cudaGetDeviceCount(&devcount),myrank,__LINE__);		
	
	if(devcount > 0)
	{	
		for(i = myrank; i < devcount; i+=comm_size)
		{
			safe_call(cudaSetDevice(i),myrank,__LINE__);		
			safe_call(cudaGetDevice(&device),myrank,__LINE__);

			if(device == i)
			{
				safe_call(cudaGetDeviceProperties(&devprop,device),myrank,__LINE__);
				strcpy(devname,devprop.name);		
				
				for(SIZE=START ; SIZE<=END; SIZE+=STEP)			
				{
					h_A = (char *) malloc(SIZE*sizeof(char));	
					h_B = (char *) malloc(SIZE*sizeof(char));		

					if(h_A==NULL || h_B==NULL)
					{
						if(myrank == 0)
							printf("Error : host memory allocation, Line : %d\n",myrank,__LINE__);
						MPI_Finalize();
						exit(-1);
					}

					safe_call(cudaMalloc((void **)&d_A, SIZE*sizeof(char)),myrank,__LINE__);
					safe_call(cudaMalloc((void **)&d_B, SIZE*sizeof(char)),myrank,__LINE__);

					fill_data(h_A,SIZE);

					safe_call(cudaEventCreate(&start),myrank,__LINE__);
					safe_call(cudaEventCreate(&stop),myrank,__LINE__);


					/************************************** Host to Device Starts ***********************************/
					safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

					safe_call(cudaMemcpy((void *)d_A, (void *)h_A, SIZE*sizeof(char), cudaMemcpyHostToDevice),myrank,__LINE__);

					safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
					safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

					safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

					time = diff*1.0e-3;	
					h2d = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
					/************************************** Host to Device Ends **************************************/



					/************************************** Device to Device Starts **********************************/
					safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

					safe_call(cudaMemcpy((void *)d_B, (void *)d_A, SIZE*sizeof(char), cudaMemcpyDeviceToDevice),myrank,__LINE__);

					safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
					safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

					safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

					time = diff*1.0e-3;	
					d2d = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
					/************************************** Device to Device Ends ************************************/



					/************************************** Device to Host Starts ************************************/	
					safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

					safe_call(cudaMemcpy((void *)h_B, (void *)d_B, SIZE*sizeof(char), cudaMemcpyDeviceToHost),myrank,__LINE__);

					safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
					safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

					safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

					time = diff*1.0e-3;	
					d2h = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
					/************************************** Device to Host Ends **************************************/	


					printf("\n\
						Process %d : %s\n \
						Device %d : %s\n \ 
						Size of Data : %dB\n \
						Host to Device : %fMB/s\n \
						Device to Device : %fMB/s\n \ 
						Device to Host : %fMB/s\n", \
						myrank,myname,device,devname,SIZE,h2d,d2d,d2h);

					safe_call(cudaFree(d_A),myrank,__LINE__);
					safe_call(cudaFree(d_B),myrank,__LINE__);

					free(h_A);
					free(h_B);

					safe_call(cudaEventDestroy(start),myrank,__LINE__);	
					safe_call(cudaEventDestroy(stop),myrank,__LINE__);
				}
			}
		}
	}
	else
	{
		if(myrank == 0)
			printf("No devices found.\n");
	}

	MPI_Finalize();
	
	return 0;
}
