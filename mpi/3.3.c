/*
Example 3.3 

Write MPI program to send a message from process with rank 0 
to process with rank 1 and process with rank 1 to send a 
message to process with 0 on a Parallel Computing System 
(Use MPI point-to-point blocking communication library calls 
and order these library calls to avoid deadlock).( Assignment)
*/

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "mpi.h"

#define SIZE 80

int main(int argc, char **argv)
{
// Variable initialization
	int MyRank, NumProcs, tag=29;
	int Root=0;
	char msg[SIZE];
	char recv[SIZE];
	MPI_Status status;

// MPI implementation
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
	MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);

// Check for number of processes
	if(NumProcs!=2)
	{
		printf("Number of Processes must be 2.\n");
		MPI_Finalize();
		exit(0);
	}

// Rank 0
	if(MyRank==0)
	{
		strcpy(msg,"Message from Rank 0");	
		printf("Sending message to 1 from 0\n");
		MPI_Send(msg,80,MPI_CHAR,1,tag,MPI_COMM_WORLD);
		printf("Receiving message from 1 to 0\n");
		MPI_Recv(recv,80,MPI_CHAR,1,tag,MPI_COMM_WORLD,&status);
		printf("Rank 0 received message : %s\n",recv);
	}
// Rank 1
	else
	{	
		strcpy(msg,"Message from Rank 1");
		printf("Sending message to 0 from 1\n");
		MPI_Send(msg,80,MPI_CHAR,0,tag,MPI_COMM_WORLD);
		printf("Receiving message from 0 to 1\n");
		MPI_Recv(recv,80,MPI_CHAR,0,tag,MPI_COMM_WORLD,&status);
		printf("Rank 1 received message : %s\n",recv);
	}

	MPI_Finalize();
}
