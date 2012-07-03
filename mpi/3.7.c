/*
Example 3.7 

Write MPI program to send a message from process with rank 0 
to process with rank 1, process with rank 1 sends a message 
to process with 2, and process with rank 2 sends a message to 
process with 3 and so on ... on a parallel computing system. 
(Use MPI MPI_Sendrecv and MPI_Sendrecv_replace communication 
library calls and order these library calls). ( Assignment)
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
	MPI_Status status;

// MPI implementation
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
	MPI_Comm_size(MPI_COMM_WORLD,&NumProcs);

// Check for number of processes
	if(NumProcs<2)
	{
		printf("Number of Processes must be greater than or equal to 2.\n");
		MPI_Finalize();
		exit(0);
	}

// Send and Receive messages
	sprintf(msg,"Message from Rank %d",MyRank);
	if(MyRank==0)
	{
		printf("Sending message to 1 from 0\n");
		MPI_Send(msg,80,MPI_CHAR,1,tag,MPI_COMM_WORLD);
	}
	else if(MyRank<NumProcs-1)
	{	
		MPI_Sendrecv_replace(msg,80,MPI_CHAR,MyRank+1,tag,MyRank-1,tag,MPI_COMM_WORLD,&status);
		printf("Rank %d received message : %s\n",MyRank,msg);
	}
	else
	{
		printf("Receiving message from %d to %d\n",MyRank-1,MyRank);
		MPI_Recv(msg,80,MPI_CHAR,MyRank-1,tag,MPI_COMM_WORLD,&status);
		printf("Rank %d received message : %s\n",MyRank,msg);
	}

	MPI_Finalize();
}
