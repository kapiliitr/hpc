/*

Objective : Write a MPI program to calculate infinity norm of a matrix using row wise block-striped partitioning.

Author : Kapil Agarwal

Date : 24 May 2012

Input : infndata.inp

*/

#include<stdio.h>
#include "mpi.h"
#include<stdlib.h>

#define root 0

int main(int argc, char **argv)
{
// Declaration of variables
	int myRank, Size, numRows, numCols, i, j, scatterSize; 
	FILE *f;
	float **input, **output, max, sum, infNorm;
	MPI_Status status;
	MPI_Datatype rowtype;

// MPI Initialisation
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &Size);
	
	if(myRank==0)
	{	
// Input data
		f = fopen("data/infndata.inp", "r");
		fscanf(f, "%d %d\n", &numRows, &numCols);

		input = (float **)malloc(numRows*sizeof(float *));
		for(i = 0; i<numRows; i++)
			input[i] = (float *)malloc(numCols*sizeof(float));

		for(i=0; i<numRows; i++)
			for(j=0; j<numCols; j++)
				fscanf(f, "%f", &input[i][j]);
		
// Display input data
		for(i=0; i<numRows; i++)
		{
			for(j=0; j<numCols; j++)
			{
				printf("%3f ",myRank,i,j,input[i][j]);
			}	
			printf("\n");
		}			

		fclose(f);				
	}
	
// Broadcast number of rows
	MPI_Bcast(&numRows, 1, MPI_INT, root, MPI_COMM_WORLD);	

// Check if number of rows is multiple of number of processes and \
number of rows is greater than or equal to number of processes
	if(numRows%Size !=0 || numRows<Size)
	{
		MPI_Finalize();
		if(myRank==0)
		{
			printf("Error.....aborting\n");
		}
		exit(0);
	}

// Broadcast number of colums
	MPI_Bcast(&numCols, 1, MPI_INT, root, MPI_COMM_WORLD);	

// New datatype to send rows
	MPI_Type_contiguous(numCols, MPI_FLOAT, &rowtype);
	MPI_Type_commit(&rowtype);

// Scatter rows to each process
	scatterSize = numRows/Size;
	output = (float **)malloc(scatterSize*sizeof(float *));
	for(i = 0; i<scatterSize; i++)
		output[i] = (float *)malloc(numCols*sizeof(float));

	MPI_Scatter(&input[0][0], scatterSize, rowtype, &output[0][0], scatterSize, rowtype, root, MPI_COMM_WORLD);

// Display received data for each process	
	for(i=0; i<scatterSize; i++)
	{
		for(j=0; j<numCols; j++)
		{
			printf("%f ",myRank,i,j,output[i][j]);
		}	
		printf("\n");
	}		

// Find out out max sum of absolute terms in each row received	
	max = 0;
	for(i=0; i<scatterSize; i++)
	{
		sum = 0;
		for(j=0; j<numCols; j++)
		{
			sum+=(output[i][j]>=0)?(output[i][j]):(0-output[i][j]);
			printf("%f ",output[i][j]);
		}	
		max = (max<sum)?sum:max;
		printf("Rank=%d max=%f\n",myRank,max);
	}		
	
// Wait till all processes have found respective max values
	MPI_Barrier(MPI_COMM_WORLD);

// Find out the Infinity Norm
	MPI_Reduce(&max, &infNorm, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

// Display Infinity Norm
	if(myRank==0)
	{
		printf("Infinity Norm = %f\n",infNorm);
	}	

	MPI_Finalize();
}
