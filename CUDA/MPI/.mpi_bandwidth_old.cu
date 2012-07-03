#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<mpi.h>
#include<string.h>
#include<sys/time.h>

#define SIZE 1024

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
	int 		comm_size, myrank;
	MPI_Status 	status;	
	char 		myname[MPI_MAX_PROCESSOR_NAME];
	int 		namelen, devcount, device;
	char		*sendbuf, *recvbuf, devname[256];
	int 		i, recvsize, sendsize, *sendcount, *displacement;
	int 		tempdisp;
	cudaDeviceProp	devprop;
	char 		*h_A, *h_B;
	char 		*d_A, *d_B;
	cudaEvent_t 	start, stop;
	double 		time, h2d, d2d, d2h;	
	float 		diff;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	MPI_Get_processor_name(myname, &namelen);
	myname[namelen++] = (char)0;         
	
	safe_call(cudaGetDeviceCount(&devcount),myrank,__LINE__);		

	if(myrank == 0)
	{
		sendsize = devcount*SIZE*sizeof(char);
		sendbuf = (char *) malloc(sendsize);
		fill_data(sendbuf,devcount*SIZE);
	}
	
	sendcount = (int *) malloc(comm_size*sizeof(int));
	displacement = (int *) malloc(comm_size*sizeof(int));
	
	tempdisp = 0;
	for(i = 0; i < ((comm_size<devcount)?comm_size:devcount); i++)
	{	
		displacement[i] = tempdisp;

		if(devcount%comm_size == 0)
			sendcount[i] = (devcount/comm_size)*SIZE;
		else
		{
			if(myrank < (devcount % comm_size))
				sendcount[i] = (devcount/comm_size + 1)*SIZE;
			else	
				sendcount[i] = (devcount/comm_size)*SIZE;
		}

		tempdisp += sendcount[i];
	}

	if(devcount % comm_size == 0)	
		recvsize = (devcount/comm_size)*SIZE;
	else
	{
		if(myrank < (devcount % comm_size))
			recvsize = (devcount/comm_size + 1)*SIZE;
		else	
			recvsize= (devcount/comm_size)*SIZE;
	}
	recvbuf = (char *) malloc(recvsize*sizeof(char));		
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Scatterv(&sendbuf,sendcount,displacement,MPI_CHAR,&recvbuf,recvsize,MPI_CHAR,0,MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);

	safe_call(cudaEventCreate(&start),myrank,__LINE__);
	safe_call(cudaEventCreate(&stop),myrank,__LINE__);

	for(i = myrank; i < devcount; i+=comm_size)
	{
		safe_call(cudaSetDevice(i),myrank,__LINE__);		
		safe_call(cudaGetDevice(&device),myrank,__LINE__);
		
		if(device == i)
		{
			safe_call(cudaGetDeviceProperties(&devprop,device),myrank,__LINE__);
			strcpy(devname,devprop.name);		
			
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

			memcpy(h_A,&recvbuf[((i-myrank)/comm_size)*SIZE],SIZE*sizeof(char));			
			
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

			printf("Device %d : %s\n \
				Host to Device : %fMB/s\n \
				Device to Device : %fMB/s\n \
				Device to Host : %fMB/s\n \
				",device,devname,h2d,d2d,d2h);
			
			safe_call(cudaFree(d_A),myrank,__LINE__);
			safe_call(cudaFree(d_B),myrank,__LINE__);

			free(h_A);
			free(h_B);
		}
	}
	
	safe_call(cudaEventDestroy(start),myrank,__LINE__);	
	safe_call(cudaEventDestroy(stop),myrank,__LINE__);

	MPI_Finalize();
	
	return 0;
}
