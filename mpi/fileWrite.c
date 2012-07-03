#include<stdio.h>
#include<stdlib.h>
#include "mpi.h"

#define BUFSIZE 80
#define NAMESIZE 128

int main(int argc, char **argv)
{
	int 		myRank, numProcs, err, i;	
	MPI_Status	status; 
	char 		*buf, *filename;	
	MPI_File 	file;
	double          time_start, time_end, time_lap;
	FILE 		*pfile;
		
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	
	if(numProcs>8)
	{	
		if(myRank==0)
			printf("Number of processes should be less than equal to 8.\n");
		MPI_Finalize();
		exit(-1);
	}	
	
	filename = (char *) malloc(NAMESIZE*sizeof(char));
	buf = (char *) malloc(BUFSIZE*sizeof(char));

	sprintf(filename, "out_par.out");
	sprintf(buf, "\nHello World from process %d.", myRank);	
		
	time_lap = 0.0;	

	time_start = MPI_Wtime();	

	err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);	
	if(err!=MPI_SUCCESS)
	{
		if(myRank==0)
			printf("Error in File open. Error : %d\n",err);
		MPI_Finalize();
		exit(1);
	}

	err = MPI_File_set_view(file, myRank*BUFSIZE*sizeof(char), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);	
	if(err!=MPI_SUCCESS)
	{
		if(myRank==0)
			printf("Error in File set view. Error : %d\n",err);
		MPI_Finalize();
		exit(1);
	}

	MPI_File_write(file, buf, BUFSIZE, MPI_CHAR, &status);
	if(err!=MPI_SUCCESS)
	{
		if(myRank==0)
			printf("Error in File write. Error : %d\n",err);
		MPI_Finalize();
		exit(1);
	}
	
	MPI_File_close(&file);
	if(err!=MPI_SUCCESS)
	{
		if(myRank==0)
			printf("Error in File close. Error : %d\n",err);
		MPI_Finalize();
		exit(1);
	}

	MPI_Barrier(MPI_COMM_WORLD);
		
	time_end = MPI_Wtime();

	time_lap += time_end - time_start;
	
	if(myRank==0)
	{
		printf("Parallel Latency = %15.6lf\n",time_lap);
	
		sprintf(filename, "out_ser.out");
	
		time_lap = 0.0;
		time_start = MPI_Wtime();
		pfile = fopen(filename, "w");
		for(i=0;i<numProcs;i++)
			fprintf(pfile, "\nHello World from process %d.", i);
		fclose(pfile);	
		time_end = MPI_Wtime();
		time_lap += time_end - time_start;
		printf("Serial Latency = %15.6lf\n",time_lap);
	}
	
	MPI_Finalize();
	
	free(buf);
	free(filename);
	
	return 0;
}
