/*
Author  : Kapil Agarwal
Date    : 10 July 2012
Compile : make new_mpi_matmatmult
Help    : mpirun -n <no of processes> -host <host ip> ./new_mpi_matmatmult <matrix size> <work group size>
*/

#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<string.h>
#include<CL/cl.h>
#include<mpi.h>

#define ERROR 1.0e-12

const char * kernelString = "mpi_matmatmult_kernel.cl";

cl_platform_id SelectPlatform()
{
    cl_int errNum;
    cl_uint numPlatforms, numDevices;
    cl_platform_id *PlatformIds, selectedPlatformId;

    // Select an OpenCL platform
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
	printf("Failed to find any OpenCL platforms.\n");
        return NULL;
    }
    PlatformIds = (cl_platform_id *) malloc(sizeof(cl_platform_id)*numPlatforms);
    errNum = clGetPlatformIDs(numPlatforms, PlatformIds, NULL);
    if (errNum != CL_SUCCESS)
    {
	printf("Failed to get OpenCL platform IDs.\n");
        return NULL;
    }
 
    for(int i=0; i<numPlatforms; i++)
    {    
	errNum = clGetDeviceIDs( PlatformIds[i], CL_DEVICE_TYPE_GPU, 0, 0, &numDevices);
	if(errNum == CL_SUCCESS)
	{
		selectedPlatformId = PlatformIds[i];
		break;
	}
    }
    if(errNum != CL_SUCCESS)
    {
	printf("Failed to find any GPU devices.\n");
	return NULL;
    }
   
    return selectedPlatformId;
}

cl_context CreateContext(cl_platform_id selectedPlatformId, cl_uint num_devices, cl_device_id *device, cl_uint displacement)
{
    cl_int errNum;
    cl_context context = NULL;
    cl_device_id *devices;

    devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
    for(int i=0; i<num_devices; i++)
	devices[i] = device[i+displacement];

    // Create an OpenCL context on the platform
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)selectedPlatformId,
        0
    };
    context = clCreateContext(contextProperties, num_devices, devices, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
	    printf("Failed to create an OpenCL GPU context.\n");
	    free(devices);
	    return NULL;
    }

    free(devices);

    return context;
}

cl_uint GetDeviceCount(cl_platform_id platform)
{
    cl_int errNum;
    cl_uint  numDevices;

    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices <= 0)
    {
        printf("No devices available.\n");
	return 0;
    }

    return numDevices;
}

cl_device_id* GetDeviceIds(cl_platform_id platform, cl_uint numDevices)
{
    cl_int errNum;
    cl_device_id *devices;
    
    devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Failed to get device IDs\n");
	free(devices);
        return NULL;
    }

    return devices;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device, int devnum)
{
    cl_command_queue commandQueue = NULL;

    commandQueue = clCreateCommandQueue(context, device, 0, NULL);
    if (commandQueue == NULL)
    {
        printf("Failed to create commandQueue for device %d.\n", devnum);
        return NULL;
    }

    return commandQueue;
}

char* read_source(const char *source_path)
{
        FILE    * pFile;
        char    *source_string;
        size_t  len;

        pFile = fopen (source_path , "r");
        if (pFile == 0)
        {
                printf("Error opening kernel source\n");
                return NULL;
        }
        else
        {
                fseek(pFile, 0, SEEK_END);
                len = ftell(pFile);
                rewind(pFile);
                source_string = (char *)malloc( len + 1);

                if( fread( source_string, 1, len, pFile) != len )
                {
                        printf("\n\t Error : Fail to read file ");
                        return 0;
                }

                source_string[len+1]='\0';
                fclose (pFile);
                return source_string;
        }
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    char * kernel_source = read_source(fileName);

    program = clCreateProgramWithSource(context,1,(const char **) &kernel_source,NULL,NULL);
    if (program == NULL)
    {
        printf("Failed to create CL program from source.\n");
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    switch(errNum)
    {
	case CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
	case CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
	case CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
	case CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
	case CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
	case CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
	case CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
	case CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
    }
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        printf("Error in kernel: \n");
        printf("%s\n",buildLog);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

void fill_array(double *arr, int len)
{
        int i;

        for(i=0;i<(len*len);i++)
                arr[i] = drand48();
}

int check(double *a, double *b, double *x, int len)
{
        int i,j,k;
        double result;
        double *c;
        c = (double *) malloc(sizeof(double)*len*len);
        for(i=0;i<len;i++)
        {
                for(j=0;j<len;j++)
                {
                        result = 0.0;
                        for(k=0;k<len;k++)
                                result += (a[i*len+k] * b[k*len+j]);
                        c[i*len+j] = result;
                        if(fabs(c[i*len+j]-x[i*len+j])>ERROR)
                        {
                                free(c);
                                return 0;
                        }
                }
        }
        free(c);
        return 1;
}

bool CreateMemObjects(cl_context context, cl_mem memObjects[3], double *a, double *b, int SIZE)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE * SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * SIZE * SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * SIZE * SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        printf("Error creating memory objects.\n");
        return false;
    }

    return true;
}

void Cleanup(cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);
}

int main(int argc, char *argv[])
{
	char			errString[120];
	int 			i, j, SIZE;
	int 			commSize, myRank;
	int			newRank, cntWorkingProc, *WorkingProc;
	int	 		PlatformStatus=1, ContextStatus=1, DeviceStatus=1, allStatus=1;
	int 			*allPlatformStatus, *allContextStatus, *allDeviceStatus;
	double                  *h_A, *h_B, *h_C, gflops;
	MPI_Status		status;
	cl_int 			err;
	cl_uint 		numDevices, numDevicesProc, displacement=0;
	cl_ulong 		start, end;
	cl_command_queue 	cmdQueue;
	cl_platform_id 		PlatformId;
	cl_context		context;
	cl_device_id 		*devices;
	cl_program		program;
	cl_kernel		kernel;
	cl_event		event;
	cl_mem			memObjects[3]={0,0,0};
	size_t                  globalWorkSize[2], localWorkSize[2];
	double 			time_start, time_end;
	struct 	timeval 	tv;
	struct 	timezone 	tz;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&commSize);
	MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
	
	if(commSize > 8 || commSize<=0)
	{
		if(myRank == 0)
			printf("Number of processes should be between 1 and 8.\n");
		MPI_Finalize();
		exit(-1);
	}

	if(argc==3)
	{       
		SIZE = atoi(argv[1]);
		if(atoi(argv[2]) > 32)
		{
			if(myRank == 0)
				printf("Maximum work group size is 32\n");
			MPI_Finalize();
			exit(-1);
		}
		if(SIZE%atoi(argv[2]) != 0)
		{
			if(myRank == 0)
				printf("Work group size should be multiple of Matrix dimension\n");
			MPI_Finalize();
			exit(-1);
		}
	}
	else
	{
		if(myRank == 0)
			printf("Syntax: mpirun -n <no of processes> -host <host ip> ./mpi_matmatmult_naive <matrix size> <work group size>\n");
		MPI_Finalize();
		exit(-1);
	}

	localWorkSize[0] = atoi(argv[2]);
	localWorkSize[1] = atoi(argv[2]);
	globalWorkSize[0] = SIZE;
	globalWorkSize[1] = SIZE;

	assert ((h_A = (double *) malloc(sizeof(double) * SIZE * SIZE)) != NULL);
	assert ((h_B = (double *) malloc(sizeof(double) * SIZE * SIZE)) != NULL);
	assert ((h_C = (double *) malloc(sizeof(double) * SIZE * SIZE)) != NULL);

	fill_array(h_A,SIZE);
	fill_array(h_B,SIZE);

	allPlatformStatus = (int *) malloc(sizeof(int) * commSize);
	allContextStatus = (int *) malloc(sizeof(int) * commSize);
	allDeviceStatus = (int *) malloc(sizeof(int) * commSize);

	PlatformStatus = 1; DeviceStatus = 1;
	PlatformId = SelectPlatform();
	if(PlatformId == NULL)
		PlatformStatus = 0;
	if(PlatformStatus != 0)
	{
		numDevices = GetDeviceCount(PlatformId);		
		if(numDevices == 0)
			DeviceStatus = 0;
		if(DeviceStatus != 0)
		{
			devices = (cl_device_id *) malloc(sizeof(cl_device_id) * numDevices);
			devices = GetDeviceIds(PlatformId,numDevices);
			if(devices == NULL)
				DeviceStatus = 0;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	WorkingProc = (int *) malloc(sizeof(int) * commSize);

	MPI_Gather(&PlatformStatus, 1, MPI_INT, allPlatformStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&DeviceStatus, 1, MPI_INT, allDeviceStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(myRank == 0)
	{
		cntWorkingProc=0;
		for(i=0; i<commSize; i++)
		{
			allStatus = 1;
			if(allPlatformStatus[i] != 0)
			{
				if(allDeviceStatus[i] == 0)
				{
					sprintf(errString,"Device Status is 0 on Process %d.",i);
					allStatus = 0;
				}
			}
			else
			{
				sprintf(errString,"Platform Status is 0 on Process %d.",i);
				allStatus = 0;
			}
			WorkingProc[i] = allStatus;
			if(allStatus == 1)
				cntWorkingProc++;
			else
				printf("%s Process %d not in use.\n",errString,i);		
		}	
	}

	MPI_Barrier(MPI_COMM_WORLD);
		
	MPI_Bcast(&cntWorkingProc, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(WorkingProc, commSize, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	if(WorkingProc[myRank] == 1)
	{
		newRank = myRank;
		for(i=0; i<myRank; i++)	
		{
			if(WorkingProc[i] == 0)
				newRank--;
		}
		
		if(newRank < numDevices)
		{	
			if(numDevices % cntWorkingProc == 0)
			{
				numDevicesProc = numDevices/cntWorkingProc;
				displacement = numDevicesProc * newRank;
			}
			else
			{
				if(newRank < (numDevices % cntWorkingProc))
				{
					numDevicesProc = numDevices/cntWorkingProc + 1;
					displacement = numDevicesProc * newRank;		
				}
				else
				{
					numDevicesProc = numDevices/cntWorkingProc;
					displacement = (numDevicesProc + 1) * (numDevices % cntWorkingProc) + numDevicesProc * (newRank - (numDevices % cntWorkingProc));
				}
			}

			context = CreateContext(PlatformId, numDevicesProc, devices, displacement);
			if(context == NULL)
				ContextStatus = 0;
			if(ContextStatus == 0)
			{
				printf("Context could not be created by Process %d.\n",myRank);

			}
			else
			{
				for(i=displacement; i<(numDevicesProc+displacement); i++)
				{
					cmdQueue = CreateCommandQueue(context, devices[i], i);
					if(cmdQueue == NULL)
					{
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}
					program = CreateProgram(context, devices[i], kernelString);
					if(program == NULL)
					{
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}
					if (!CreateMemObjects(context, memObjects, h_A, h_B, SIZE))
					{
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}

					kernel = clCreateKernel(program,"matmatmult",&err);
					if (kernel == NULL)
					{
						printf("Failed to create kernel.\n");
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}

					err = clSetKernelArg(kernel,0,sizeof(cl_mem),&memObjects[0]);
					err |= clSetKernelArg(kernel,1,sizeof(cl_mem),&memObjects[1]);
					err |= clSetKernelArg(kernel,2,sizeof(cl_mem),&memObjects[2]);
					err |= clSetKernelArg(kernel,3,sizeof(int),(void *)&SIZE);
					if(err != CL_SUCCESS)
					{
						printf("Error setting kernel arguments.\n");
						Cleanup(cmdQueue, program, kernel, memObjects);

						continue;
					}

					gettimeofday(&tv, &tz);

					err = clEnqueueNDRangeKernel(cmdQueue,kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,&event);

					gettimeofday(&tv, &tz);

					time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
					time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					if (err != CL_SUCCESS)
					{
						printf("Error queuing kernel for execution.\n");
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}

					err = clFinish(cmdQueue);
					if (err != CL_SUCCESS)
					{
						printf("Error in finishing command queue.\n");
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}

					clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
					clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

					gflops = 1.0e-9 * ( 2.0 * SIZE * SIZE * SIZE )/(float)(time_end - time_start);
					//gflops = ( 2.0 * SIZE * SIZE * SIZE )/(float)(1.0e-9 * (end - start));

					err = clEnqueueReadBuffer(cmdQueue,memObjects[2],CL_TRUE,0,sizeof(double)*SIZE*SIZE,h_C,0,NULL,NULL);
					if (err != CL_SUCCESS)
					{
						printf("Error reading result buffer.\n");
						Cleanup(cmdQueue, program, kernel, memObjects);
						continue;
					}

					if(check(h_A,h_B,h_C,SIZE))
						printf("----------------------\nResult verification success\nProcess : %d\nDevice : %d\nGflops : %f\n",myRank,i,gflops);
					else
						printf("----------------------\nResult verification failed\nProcess : %d\tDevice : %d\n",myRank,i);

					Cleanup(cmdQueue, program, kernel, memObjects);
				}

				clReleaseContext(context);
			}	
		}
	}	
	
	MPI_Barrier(MPI_COMM_WORLD);

	free(WorkingProc);
	free(h_A);
	free(h_B);
	free(h_C);
	free(devices);
	if(myRank == 0)
	{
		free(allPlatformStatus);
		free(allContextStatus);
		free(allDeviceStatus);
	}

	MPI_Finalize();

	return 0;
}
