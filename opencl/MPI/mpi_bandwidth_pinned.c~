/*
Author  : Kapil Agarwal
Date    : 10 July 2012
Compile : make mpi_bandwidth_pinned
Help    : mpirun -n <no of processes> -host <host ip> ./mpi_bandwidth_pinned <matrix size> <work group size>
*/

#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<string.h>
#include<CL/cl.h>
#include<mpi.h>

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

void fill_array(unsigned char *arr, int len)
{
	int i;
	
	for(i=0;i<len;i++)
	{
		arr[i] = (unsigned char)(i & 0xff);
	}
}

bool check(unsigned char *a, unsigned char *b, int len)
{
        int i;

	for(i=0; i<len; i++)
		if(a[i] != b[i])
			return false;
		
	return true;
}

void Cleanup(cl_command_queue commandQueue, cl_mem cmPinnedBufIn, cl_mem cmPinnedBufOut, cl_mem cmDevBufIn, cl_mem cmDevBufOut)
{
    if(cmPinnedBufIn != 0)
	clReleaseMemObject(cmPinnedBufIn);
	
    if(cmPinnedBufOut != 0)
	clReleaseMemObject(cmPinnedBufOut);

    if(cmDevBufIn != 0)
	clReleaseMemObject(cmDevBufIn);

    if(cmDevBufIn != 0)
	clReleaseMemObject(cmDevBufIn);

    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
}

int main(int argc, char *argv[])
{
	char			errString[120];
	int 			i, j, memSize;
	int 			commSize, myRank;
	int			newRank, cntWorkingProc, *WorkingProc;
	int	 		PlatformStatus=1, ContextStatus=1, DeviceStatus=1, allStatus=1;
	int 			*allPlatformStatus, *allContextStatus, *allDeviceStatus;
	double                  h2d, d2d, d2h;
	MPI_Status		status;
	cl_int 			err;
	cl_uint 		numDevices, numDevicesProc, displacement=0;
	cl_ulong 		start, end;
	cl_platform_id 		PlatformId;
	cl_device_id 		*devices;
	cl_event		event_write, event_read, event_copy;
	double 			time_start, time_end;
	struct 	timeval 	tv;
	struct 	timezone 	tz;
	/*
	   Declare cl_mem buffer objects for the pinned host memory and the GPU 
	   device GMEM, respectively, and standard pointers to reference pinned host 
	   memory
	 */
	cl_command_queue 	cqCommandQue;
	cl_context 		cxGPUContext; 
	cl_mem 			cmPinnedBufIn = NULL, cmPinnedBufOut = NULL, cmDevBufIn = NULL, cmDevBufOut = NULL; 
	unsigned char		*cDataIn = NULL; 
	unsigned char		*cDataOut = NULL; 

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

	if(argc==2)
	{       
		memSize = atoi(argv[1]);
	}
	else
	{
		if(myRank == 0)
			printf("Syntax: mpirun -n <no of processes> -host <host ip> ./mpi_bandwidth_pinned <transfer size>\n");
		MPI_Finalize();
		exit(-1);
	}

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

			cxGPUContext = CreateContext(PlatformId, numDevicesProc, devices, displacement);
			if(cxGPUContext == NULL)
				ContextStatus = 0;
			if(ContextStatus == 0)
			{
				printf("Context could not be created by Process %d.\n",myRank);
			}
			else
			{
				for(i=displacement; i<(numDevicesProc+displacement); i++)
				{
					cqCommandQue = CreateCommandQueue(cxGPUContext, devices[i], i);
					if(cqCommandQue == NULL)
					{
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
					
					/*
					   Allocate cl_mem buffer objects for the pinned host memory and the GPU 
					   device GMEM, respectively
					 */
					cmPinnedBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, NULL);
					cmPinnedBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, NULL); 
					cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, memSize, NULL, NULL);
					cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, memSize, NULL, NULL);

					if (cmPinnedBufIn == NULL || cmPinnedBufOut == NULL || cmDevBufIn == NULL || cmDevBufOut == NULL)
					{
						printf("Error creating memory objects.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}

					/*
					   Map standard pointer to reference the pinned host memory input and output 
					   buffers with standard pointers
					 */
					cDataIn = (unsigned char*)clEnqueueMapBuffer(cqCommandQue, cmPinnedBufIn, CL_TRUE, CL_MAP_WRITE, 0, memSize, 0, NULL, NULL, NULL); 
					cDataOut = (unsigned char*)clEnqueueMapBuffer(cqCommandQue, cmPinnedBufOut, CL_TRUE, CL_MAP_READ, 0, memSize, 0, NULL, NULL, NULL); 										

					/*
					   Initialize or update the pinned memory content, using the standard host pointer 
					   and standard host code
					 */
					fill_array(cDataIn,memSize);

					/*
					   Write data from pinned host memory to the GPU device GMEM
					 */
					gettimeofday(&tv, &tz);
					time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					err = clEnqueueWriteBuffer(cqCommandQue, cmDevBufIn, CL_FALSE, 0, memSize, cDataIn, 0, NULL, &event_write);
					gettimeofday(&tv, &tz);
					time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					if (err != CL_SUCCESS)
					{
						printf("Error in writing to buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
					clFinish(cqCommandQue);
					if (err != CL_SUCCESS)
					{
						printf("Error in finishing command queue after writing buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
					

					clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					//h2d = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));
					h2d = ((double)memSize * sizeof(unsigned char))/((time_end - time_start) * (double)(1 << 20));

					/*
					   Copy data from GPU device GMEM to the GPU device GMEM
					 */
					gettimeofday(&tv, &tz);
					time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					err = clEnqueueCopyBuffer(cqCommandQue, cmDevBufIn, cmDevBufOut, 0, 0, memSize, 0, NULL, &event_copy);
					gettimeofday(&tv, &tz);
					time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					if (err != CL_SUCCESS)
					{
						printf("Error in copying buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
					clFinish(cqCommandQue);
					if (err != CL_SUCCESS)
					{
						printf("Error in finishing command queue after copying buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}


					clGetEventProfilingInfo(event_copy, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_copy, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					//d2d = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));
					d2d = ((double)memSize * sizeof(unsigned char))/((time_end - time_start) * (double)(1 << 20));

					/*
					   Read data from GPU device GMEM to pinned host memory
					 */
					gettimeofday(&tv, &tz);
					time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					err = clEnqueueReadBuffer(cqCommandQue, cmDevBufOut, CL_TRUE, 0, memSize, cDataOut, 0, NULL, &event_read);
					gettimeofday(&tv, &tz);
					time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

					if (err != CL_SUCCESS)
					{
						printf("Error in reading from buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
					
					err = clFinish(cqCommandQue);
					if (err != CL_SUCCESS)
					{
						printf("Error in finishing command queue after reading buffer.\n");
						Cleanup(cqCommandQue, cmPinnedBufIn, cmPinnedBufOut, cmDevBufIn, cmDevBufOut);
						continue;
					}
	
					clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					//d2h = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));
					d2h = ((double)memSize * sizeof(unsigned char))/((time_end - time_start) * (double)(1 << 20));
		
					/*
					   Check if data has been transferred successfully
					 */
					if(check(cDataOut,cDataIn,memSize))
						printf("--------------------------\nProcess : %d\nDevice : %d\nHost to Device : %f MB/s\nDevice to Device : %f MB/s\nDevice to Host : %f MB/s\n",myRank,i,h2d,d2d,d2h);				
					else
						printf("--------------------------\nProcess : %d\nDevice : %d\nData transfer failed\n",myRank,i);				
					
					clReleaseMemObject(cmPinnedBufIn);
                                        clReleaseMemObject(cmPinnedBufOut);
                                        clReleaseMemObject(cmDevBufIn);
                                        clReleaseMemObject(cmDevBufOut);
                                        clReleaseCommandQueue(cqCommandQue);
				}

				if(cxGPUContext != 0)
					clReleaseContext(cxGPUContext);
			}	
		}
	}	

	MPI_Barrier(MPI_COMM_WORLD);

	free(WorkingProc);
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
