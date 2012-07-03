/*

Objective : Write a MPI program to calculate the transpose of a matrix using block checkerboard partitioning and MPI Cartesian topology.

Author : Kapil Agarwal

Date : 31 May 2012

Input : data/transpose.inp

*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<mpi.h>
#include<math.h>

int main(int argc, char *argv[])
{
    int         numProcs, p_proc, myRank, newRank, irow, icol, ProcId, Global_Row_Index, Global_Col_Index, Local_Index;
    int         Root = 0, rows, cols, *MatrixSize, rowsBlock, colsBlock;
    int         i, j, index, tmp, FileStatus = 1;
    float       **arr, **trans; 
    float       *Mat, *blockMat, *blockMatTrans, *MatTrans;
    MPI_Comm    new_comm;
    int         ndims, reorder, *periods, *dimsize, *coords;
    MPI_Status  status;   
    FILE        *fp;
    char        *fileName;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if(argc!=2)
    {
        if(myRank==Root)         
            printf("Syntax: exec <fileName>\n");
        MPI_Finalize();
        exit(-1);
    }
    
    if(numProcs>8 || numProcs<2)
    {
        if(myRank==Root)         
            printf("Number of processes should be between 2 and 8\n");
        MPI_Finalize();
        exit(-1);
    }

    p_proc = (int)sqrt((double) numProcs);
    if(p_proc*p_proc != numProcs)
    {
        MPI_Finalize();
        if(myRank == 0){
            printf("Number of Processors should be perfect square\n");
        }
        exit(-1);
    }
 
    ndims = 2;
    dimsize = (int *) malloc(2*sizeof(int));
    periods = (int *) malloc(2*sizeof(int));
    dimsize[0] = p_proc;
    dimsize[1] = p_proc;
    periods[0] = 0;
    periods[1] = 0;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dimsize, periods, reorder, &new_comm);
 
    if(new_comm==MPI_COMM_NULL)
    {
        if(myRank==Root)
            printf("Could not create Cartesian Topology\n");
        MPI_Finalize();
        exit(-1);
    }
   
    coords = (int *) malloc(2*sizeof(int));
    MPI_Cart_coords(new_comm, myRank, ndims, coords);
    MPI_Cart_rank(new_comm, coords, &newRank);
       
    MPI_Barrier(MPI_COMM_WORLD);
    
    MatrixSize = (int *) malloc(2*sizeof(int));
    fileName = (char *) malloc(80*sizeof(char));

    if(newRank == Root)
    {
        sprintf(fileName,"%s",argv[1]);
        if ((fp = fopen (fileName, "r")) == NULL){
            FileStatus = 0;
        }
        
        if(FileStatus!=0)
        {
            fscanf(fp, "%d %d\n", &rows, &cols);
            MatrixSize[0] = rows;
            MatrixSize[1] = cols;

            arr = (float **) malloc(rows*sizeof(float*));
            for(i=0;i<rows;i++)
            {
                arr[i] = (float *) malloc(cols*sizeof(float));
                for(j=0;j<cols;j++)
                    fscanf(fp,"%f",&arr[i][j]);
            }
            fclose(fp);
        }
    }
      
    MPI_Bcast(&FileStatus, 1, MPI_INT, 0, new_comm);
    if(FileStatus == 0) 
    {
        if(newRank == 0) 
            printf("Can't open input file for Matrix\n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Bcast(MatrixSize, 2, MPI_INT, 0, new_comm);
    rows = MatrixSize[0];
    cols = MatrixSize[1];

    if(rows%p_proc!=0 || cols%p_proc!=0)
    {
        if(newRank==Root)         
            printf("Matrix can't be divided among processors equally\n");
        MPI_Finalize();
        exit(-1);
    }
    
    rowsBlock = rows/p_proc;
    colsBlock = cols/p_proc;
           
    Mat = (float *) malloc(rows*cols*sizeof(float));
    blockMat = (float *) malloc(rowsBlock*colsBlock*sizeof(float));
    
    if(newRank==Root)
    {
        for (j = 0; j < p_proc; j++)
        {
            for (i = 0; i < p_proc; i++)
            {
                ProcId = j * p_proc + i;  
                for (irow = 0; irow < rowsBlock; irow++)
                {    
                    Global_Row_Index = i * rowsBlock + irow;
                    for (icol = 0; icol < colsBlock; icol++)
                    {
                        Local_Index      = (ProcId * rowsBlock * colsBlock) + (irow * colsBlock) + icol;
                        Global_Col_Index = j * colsBlock + icol;
                        Mat[Local_Index] = arr[Global_Row_Index][Global_Col_Index];
                    }
                }
            }
        }
    }

    MPI_Scatter (Mat, rowsBlock*colsBlock, MPI_FLOAT, blockMat, rowsBlock*colsBlock, MPI_FLOAT, 0, new_comm);

    blockMatTrans = (float *) malloc(rowsBlock*colsBlock*sizeof(float));
    
    index = 0;
    for(i=0;i<rowsBlock;i++)
        for(j=0;j<colsBlock;j++)
        {
            tmp = j*colsBlock + i;
            blockMatTrans[index++] = blockMat[tmp];
        }

    MatTrans = (float *) malloc(rows*cols*sizeof(float));

    trans = (float **) malloc(rows*sizeof(float*));
    for(i=0;i<rows;i++)
        trans[i] = (float *) malloc(cols*sizeof(float));
    
    MPI_Gather (blockMatTrans, rowsBlock*colsBlock, MPI_FLOAT, MatTrans, colsBlock*rowsBlock, MPI_FLOAT, 0, new_comm);

    if (newRank == Root) 
    {
        for (i = 0; i < p_proc; i++)
        {
            for (j = 0; j < p_proc; j++)
            {
                ProcId = i * p_proc + j;
                for (irow = 0; irow < rowsBlock; irow++)
                {
                    Global_Row_Index = i * rowsBlock + irow;
                    for (icol = 0; icol < colsBlock; icol++)
                    {
                        Local_Index = (ProcId * rowsBlock * colsBlock) + (irow * colsBlock) + icol;
                        Global_Col_Index = j * colsBlock + icol;
                        trans[Global_Row_Index][Global_Col_Index] = MatTrans[Local_Index];
                    }
                }
            }
        }

        printf ("----------MATRIX TRANSPOSE RESULTS --------------\n");
        printf(" Processor %d, Matrix : Dimension %d * %d : \n", newRank, rows, cols);
        for(irow = 0; irow < rows; irow++) 
        {
            for(icol = 0; icol < cols; icol++)
                printf ("%7.3f ", arr[irow][icol]);
            printf ("\n");
        }
        printf("\n");

        printf("Processor %d, Matrix : Dimension %d * %d : \n", newRank, rows, cols);
        for(irow = 0; irow < rows; irow++)
        {
            for(icol = 0; icol < cols; icol++)
                printf("%7.3f ",trans[irow][icol]);
            printf("\n");
        }

        for(irow=0; irow<rows; irow++)
            for(icol=0; icol<cols; icol++)
                trans[icol][irow] = arr[irow][icol];

        printf("Serial results\n");
        for(irow = 0; irow < rows; irow++)
        {
            for(icol = 0; icol < cols; icol++)
                printf("%7.3f ",trans[irow][icol]);
            printf("\n");
        }
    }

    MPI_Finalize();
}
