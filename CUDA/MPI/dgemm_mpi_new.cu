/*
Author  : Kapil Agarwal
Date    : 22 June 2012
Compile : make dgemm_mpi
Help    : mpirun -n <no of processes> -host <host ip> ./dgemm_mpi -help
*/

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<mpi.h>
#include<string.h>
#include<sys/time.h>
#include<ctype.h>
#include<math.h>
#include "cublas.h"

#define ERROR 1.0e-12

void call_finalize()
{
	MPI_Finalize();
	exit(-1);
}

void safe_call(cudaError_t ret, int myrank, int line)
{
	if(ret!=cudaSuccess)
	{
		printf("Error on Process %d at line %d : %s\n",myrank,line,cudaGetErrorString(ret));
		call_finalize();
	}
}

void mem_error(char *arrayname, int len, char *type, int myrank)
{
	printf("\nMemory not sufficient to allocate for array %s\n\tProcess : %d \n\tMemory requested = %d number of %s elements\n",arrayname, myrank, len, type);
	call_finalize();
}

void fill_data(double *arr, int len)
{
	int i;
	
	for(i=0;i<len;i++)
	{
		srand(time(NULL));
		arr[i] = drand48();
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

int check(double *a, double *b, int row, int col)
{
        int i;
        for(i=0;i<row*col;i++)
                if(fabs(a[i]-b[i])>ERROR)
                {
			return i;
                }
	return -1;
}

void printSyntax()
{
	printf("Syntax : \n\
		 mpirun -n <no of processes> -host <host ip> ./dgemm_mpi -options\n\
		\n\
		-help\n\
		-mode=MODE square(default),general\n\
		if mode=square\n\
		\t-rowA=no of rows in Square matrices\n\
		if mode=general\n\
		\t-rowA=no of rows in A\n\	
		\t-colB=no of cols in B\n\
		\t-colA=no of cols in A\n");
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

void safe_call_cublas(cublasStatus_t ret, int myrank, int line)
{
        if(ret!=CUBLAS_STATUS_SUCCESS)
        {
                printf("CUBLAS Error on Process %d at line %d : %s\n",myrank,line);
		call_finalize();
        }
}

void cpu_matmatmul(double *a, double *b, double *c, int rowA, int colB, int colA)
{
        int i,j,k;
        double result;
        for(i=0;i<rowA;i++)
                for(j=0;j<colB;j++)
                {
                        result = 0.0;
                        for(k=0;k<colA;k++)
                                result += (a[i*colA+k] * b[k*colB+j]);
                        c[i*colB+j] = result;
                }
}

void transpose(double *a, double *b, int row, int col)
{
        int i,j;

        for(i=0;i<row;i++)
                for(j=0;j<col;j++)
                        b[j*row+i] = a[i*col+j];
}

void printMat(double *mat, int row, int col)
{
	int i,j;
	
	for(i=0;i<row;i++)
	{
		for(j=0;j<col;j++)
			printf("%f ",mat[i*col+j]);
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int 		comm_size, myrank, i, j, no_of_args, valid_args, materr;
	int 		RowA, ColB, ColA;
	char 		MODE[10], temp_arg[80];
	char 		myname[MPI_MAX_PROCESSOR_NAME];
	int 		namelen, devcount, device;
	char		devname[256];
	cudaDeviceProp	devprop;
	double 		*h_A, *h_B, *h_C, *cpu_C;
	double 		*h_At, *h_Bt, *h_Ct;
	double		*d_A, *d_B, *d_C;
	cudaEvent_t 	start, stop;
	double 		time;	
	float 		diff, gflops;
	double 		alpha=1.0, beta=0.0;
	float		*sendbuf, *recvbuf;	
	int 		sendcnt, *recvcnts, *displs;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	// Default
	strcpy(MODE,"square");
	RowA = 16;
	ColB = 16;
	ColA = 16;

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

		if(no_of_args==4 && strcmp(MODE,"general")==0)
		{
			valid_args = 1;		

			if(get_cmd_arg(argc,argv,"rowA",temp_arg) == 1)
			{
				no_of_args--;
				if(isint(temp_arg))
					RowA = atoi(temp_arg);
				else
					valid_args=0;
			}

			if(get_cmd_arg(argc,argv,"colB",temp_arg) == 1)
			{
				no_of_args--;
				if(isint(temp_arg))
					ColB = atoi(temp_arg);
				else
					valid_args=0;
			}

			if(get_cmd_arg(argc,argv,"colA",temp_arg) == 1)
			{
				no_of_args--;
				if(isint(temp_arg))
					ColA = atoi(temp_arg);
				else
					valid_args=0;
			}

		}
			
		if(valid_args == 0)
		{
			if(myrank==0)
				printf("Enter valid values for number of rows and columns of the matrices.\n");
			call_finalize();
		}
	}

	if(strcmp(MODE,"square")==0)
	{
		if(get_cmd_arg(argc,argv,"rowA",temp_arg) == 1)
		{
			no_of_args--;
			if(isint(temp_arg))
			{
				RowA = atoi(temp_arg);
				ColB = RowA;
				ColA = RowA;
			}
			else
			{
				if(myrank==0)
					printf("Enter valid values for number of rows and columns of the matrices.\n");
				call_finalize();
			}
		}
	}
	
	if(no_of_args != 1)
	{
		if(myrank==0)
			printSyntax();	
		call_finalize();	
	}

	if(myrank == 0)
		printf("MODE=%s RowA=%d ColB=%d ColA=%d\n",MODE,RowA,ColB,ColA);

	MPI_Get_processor_name(myname, &namelen);
	myname[namelen++] = (char)0;         

	safe_call(cudaGetDeviceCount(&devcount),myrank,__LINE__);		

	sendcnt = (int)ceil((1.0*devcount-myrank)/comm_size);
	sendbuf = (float *) malloc(sendcnt*sizeof(float));
	recvbuf = (float *) malloc(devcount*sizeof(float));
	recvcnts = (int *) malloc(comm_size*sizeof(int));
	displs = (int *) malloc(comm_size*sizeof(int));

	i=0,j=0;
	if(devcount%comm_size)
		for(;i<(devcount%comm_size);i++)
		{
			recvcnts[i] = (devcount/comm_size)+1;
			displs[i] = j*sizeof(float);
			j += recvcnts[i];
		}
	for(;i<comm_size;i++)
	{
		recvcnts[i] = devcount/comm_size;
		displs[i] = j*sizeof(float);
		j += recvcnts[i];
	}

	if(devcount > 0)
	{			
		if(strcmp(MODE,"square") == 0 || strcmp(MODE,"general") == 0)
		{
			j=0;
	
			for(i = myrank; i < devcount; i+=comm_size)
			{
				safe_call(cudaSetDevice(i),myrank,__LINE__);		
				safe_call(cudaGetDevice(&device),myrank,__LINE__);

				if(device == i)
				{
					safe_call(cudaGetDeviceProperties(&devprop,device),myrank,__LINE__);
					strcpy(devname,devprop.name);		

					h_A =(double *)malloc(RowA*ColA*sizeof(double));
				        h_B = (double *)malloc(ColA*ColB*sizeof(double));
				        h_C = (double *)malloc(RowA*ColB*sizeof(double));

					h_At =(double *)malloc(RowA*ColA*sizeof(double));
				        h_Bt = (double *)malloc(ColA*ColB*sizeof(double));
				        h_Ct = (double *)malloc(RowA*ColB*sizeof(double));

					if(h_A==NULL)
						mem_error("h_A",RowA*ColA,"double",myrank);

					if(h_B==NULL)
						mem_error("h_B",ColA*ColB,"double",myrank);

					if(h_C==NULL)
						mem_error("h_C",RowA*ColB,"double",myrank);

					if(h_At==NULL)
						mem_error("h_At",RowA*ColA,"double",myrank);

					if(h_Bt==NULL)
						mem_error("h_Bt",ColA*ColB,"double",myrank);

					if(h_Ct==NULL)
						mem_error("h_Ct",RowA*ColB,"double",myrank);

					fill_data(h_A,RowA*ColA);
					fill_data(h_B,ColB*ColA);					
					
					transpose(h_A,h_At,RowA,ColA);
					transpose(h_B,h_Bt,ColA,ColB);
		
					safe_call(cudaEventCreate(&start),myrank,__LINE__);
					safe_call(cudaEventCreate(&stop),myrank,__LINE__);
					
					safe_call_cublas(cublasAlloc (RowA*ColA, sizeof(double), (void**)&d_A), myrank, __LINE__);
					safe_call_cublas(cublasAlloc (ColA*ColB, sizeof(double), (void**)&d_B), myrank, __LINE__);
					safe_call_cublas(cublasAlloc (RowA*ColB, sizeof(double), (void**)&d_C), myrank, __LINE__);

					safe_call_cublas(cublasSetVector (RowA*ColA, sizeof(double), h_At, 1, d_A, 1), myrank, __LINE__);
					safe_call_cublas(cublasSetVector (ColA*ColB, sizeof(double), h_Bt, 1, d_B, 1), myrank, __LINE__);

					safe_call(cudaEventRecord(start, 0), myrank, __LINE__);

					cublasDgemm('N','N',RowA,ColB,ColA,alpha,d_A,RowA,d_B,ColA,beta,d_C,RowA);				

					safe_call(cudaEventRecord (stop, 0), myrank, __LINE__);
					safe_call(cudaEventSynchronize (stop), myrank, __LINE__);

					safe_call_cublas(cublasGetVector (RowA*ColB, sizeof(double), d_C, 1, h_Ct, 1), myrank, __LINE__);

					safe_call(cudaEventElapsedTime(&diff, start, stop), myrank, __LINE__);
					time = diff *1.0e-3;

				        gflops=(1.0e-9 * (( 2.0 * RowA * ColB * ColA )/time));

				        cpu_C = (double *)malloc(RowA*ColB*sizeof(double));

					if(cpu_C==NULL)
						mem_error("cpu_C",RowA*ColB,"double",myrank);
					
					cpu_matmatmul(h_A, h_B, cpu_C, RowA, ColB, ColA);
			
					transpose(h_Ct,h_C,ColB,RowA);			

					if(materr=check(h_C,cpu_C,RowA,ColB)==-1)
						sendbuf[j++] = gflops;
					else
						sendbuf[j++] = -1;					

					safe_call_cublas(cublasFree(d_A),myrank,__LINE__);
					safe_call_cublas(cublasFree(d_B),myrank,__LINE__);
					safe_call_cublas(cublasFree(d_C),myrank,__LINE__);
				
					free(h_A);
					free(h_B);
					free(h_C);
					free(cpu_C);
					free(h_At);
					free(h_Bt);
					free(h_Ct);
					
					safe_call(cudaEventDestroy(start),myrank,__LINE__);
					safe_call(cudaEventDestroy(stop),myrank,__LINE__);
				}
			}
			
			MPI_Gatherv(sendbuf, sendcnt, MPI_FLOAT, recvbuf, recvcnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			
			if(myrank == 0)
				for(i=0;i<devcount;i++)
					if(recvbuf[i]!=-1)
						printf("\n\
							Device %d\n\
							Mode : %s\n\ 
							Dimensions of Matrix : \n\
							\t rowA : %d\n\
							\t colB : %d\n\
							\t colA : %d\n\
							Gflops\/s : %f\n",\
							i,MODE,RowA,ColB,ColA,recvbuf[i]);
					else
						printf("Error : CPU and GPU result do not match on Device:%d\n",device);	
		}
		else
		{
			if(myrank==0)
				printf("Matrix mode choices : square/general\n");
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
