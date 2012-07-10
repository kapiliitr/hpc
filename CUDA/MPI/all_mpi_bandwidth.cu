/*
Author  : Kapil Agarwal
Date    : 19 June 2012
Compile : make all_mpi_bandwidth
Help    : mpirun -n <no of processes> -host <host ip> ./all_mpi_bandwidth -help
*/

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<mpi.h>
#include<string.h>
#include<sys/time.h>
#include<ctype.h>

void call_finalize()
{
	MPI_Finalize();
	exit(-1);
}

void safe_call(cudaError_t ret, int myrank, int line)
{
	if(ret!=cudaSuccess)
	{
		if(myrank == 0)
			printf("Error at line %d : %s\n",line,cudaGetErrorString(ret));
		call_finalize();
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

int get_cmd_arg(int argc, char **arg, char *cmp, char *ret)
{
	int i,j;
	char *pch;
	i=0;
	for(j=0;j<argc;j++)
	{
		while(arg[j][i]=='-') { i++; }
		if(i!=0)
		{
			if(pch=strstr(arg[j],cmp))
			{
				if(strcmp(cmp,"help") == 0)
					return 1;
				else if(pch=strpbrk(arg[j],"="))	
				{
					strcpy(ret,pch+1);
					return 1;
				}
			}
		}	
	}
	return 0;
}

void printSyntax()
{
	printf("Syntax : \n\
		 mpirun -n <no of processes> -host <host ip> ./new_mpi_bandwidth -options\n\
		\n\
		-help\n\
		-mode=MODE pinned,pageable\n\
		-start=START\n\	
		-end=END\n\
		-step=STEP\n");
}

int isint(char *str)
{
	int i,len;

	len = strlen(str);
	
	for(i=0;i<len;i++)
		if(!isdigit(str[i]))	
			return 0;
	
	return 1;
}

int main(int argc, char *argv[])
{
	int 		comm_size, myrank, i, no_of_args, valid_args;
	int 		START, END, STEP, SIZE;
	char 		MODE[10], temp_arg[80];
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

	// Default
	strcpy(MODE,"pageable");
	START = 1024;
	END = 10240;
	STEP = 1024;

	no_of_args = argc;	

	if(get_cmd_arg(argc,argv,"help",temp_arg) == 1)
	{
		no_of_args--;
		if(myrank==0)
			printSyntax();	
		call_finalize();	
	}

	if(get_cmd_arg(argc,argv,"mode",temp_arg) == 1)
	{
		no_of_args--;
		strcpy(MODE,temp_arg);
	}

	if(no_of_args==4)
	{
		valid_args = 1;		

		if(get_cmd_arg(argc,argv,"start",temp_arg) == 1)
		{
			no_of_args--;
			if(isint(temp_arg))
				START = atoi(temp_arg);
			else
				valid_args=0;
		}

		if(get_cmd_arg(argc,argv,"end",temp_arg) == 1)
		{
			no_of_args--;
			if(isint(temp_arg))
				END = atoi(temp_arg);
			else
				valid_args=0;
		}

		if(get_cmd_arg(argc,argv,"step",temp_arg) == 1)
		{
			no_of_args--;
			if(isint(temp_arg))
				STEP = atoi(temp_arg);
			else
				valid_args=0;
		}

		if(((1.0*END-START)/STEP < 1.0))
			valid_args=0;

		if(valid_args == 0)
		{
			if(myrank==0)
				printf("Enter valid values for start, end and step.\n");
			call_finalize();
		}
	}
	else if(no_of_args != 1)
	{
		if(myrank==0)
			printSyntax();	
		call_finalize();	
	}

	if(myrank == 0)
		printf("MODE=%s START=%d END=%d STEP=%d\n",MODE,START,END,STEP);

	MPI_Get_processor_name(myname, &namelen);
	myname[namelen++] = (char)0;         

	safe_call(cudaGetDeviceCount(&devcount),myrank,__LINE__);		
	if(devcount > 0)
	{	
		if(strcmp(MODE,"pageable") == 0)
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
							call_finalize();
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
								Process %d : %s\n\
								Device %d : %s\n\
								Mode : %s\n\ 
								Size of Data : %dB\n\
								Host to Device : %fMB/s\n\
								Device to Device : %fMB/s\n\ 
								Device to Host : %fMB/s\n", \
								myrank,myname,device,devname,MODE,SIZE,h2d,d2d,d2h);

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
		else if(strcmp(MODE,"pinned") == 0)
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
						safe_call(cudaMallocHost((void **)&h_A, SIZE*sizeof(char)),myrank,__LINE__);
						safe_call(cudaMallocHost((void **)&h_B, SIZE*sizeof(char)),myrank,__LINE__);

						safe_call(cudaMalloc((void **)&d_A, SIZE*sizeof(char)),myrank,__LINE__);
						safe_call(cudaMalloc((void **)&d_B, SIZE*sizeof(char)),myrank,__LINE__);

						fill_data(h_A,SIZE);

						safe_call(cudaEventCreate(&start),myrank,__LINE__);
						safe_call(cudaEventCreate(&stop),myrank,__LINE__);	


						/************************************** Host to Device Starts ***********************************/
						safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

						safe_call(cudaMemcpyAsync((void *)d_A, (void *)h_A, SIZE*sizeof(char), cudaMemcpyHostToDevice),myrank,__LINE__);

						safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
						safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

						safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

						time = diff*1.0e-3;	
						h2d = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
						/************************************** Host to Device Ends **************************************/



						/************************************** Device to Device Starts **********************************/
						safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

						safe_call(cudaMemcpyAsync((void *)d_B, (void *)d_A, SIZE*sizeof(char), cudaMemcpyDeviceToDevice),myrank,__LINE__);

						safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
						safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

						safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

						time = diff*1.0e-3;	
						d2d = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
						/************************************** Device to Device Ends ************************************/



						/************************************** Device to Host Starts ************************************/	
						safe_call(cudaEventRecord(start, 0),myrank,__LINE__);

						safe_call(cudaMemcpyAsync((void *)h_B, (void *)d_B, SIZE*sizeof(char), cudaMemcpyDeviceToHost),myrank,__LINE__);

						safe_call(cudaEventRecord(stop, 0),myrank,__LINE__);
						safe_call(cudaEventSynchronize(stop),myrank,__LINE__);

						safe_call(cudaEventElapsedTime(&diff,start,stop),myrank,__LINE__);

						time = diff*1.0e-3;	
						d2h = ( SIZE * sizeof(char) * 2.0 ) / ( 1024 * 1024 * time ) ;	
						/************************************** Device to Host Ends **************************************/	


						printf("\n\
								Process %d : %s\n\
								Device %d : %s\n\ 
								Mode : %s\n\
								Size of Data : %dB\n\
								Host to Device : %fMB/s\n\
								Device to Device : %fMB/s\n\ 
								Device to Host : %fMB/s\n", \
								myrank,myname,device,devname,MODE,SIZE,h2d,d2d,d2h);

						safe_call(cudaFree(d_A),myrank,__LINE__);
						safe_call(cudaFree(d_B),myrank,__LINE__);
				
						safe_call(cudaFreeHost(h_A),myrank,__LINE__);
						safe_call(cudaFreeHost(h_B),myrank,__LINE__);
						
						safe_call(cudaEventDestroy(start),myrank,__LINE__);	
						safe_call(cudaEventDestroy(stop),myrank,__LINE__);
					}
				}
			}
		}
		else
		{
			if(myrank==0)
				printf("Memory mode choices : pinned/pageable\n");
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
