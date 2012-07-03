#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<string.h>
#include<CL/cl.h>

void check_opencl(cl_int err, int line)
{
	switch (err) 
	{
		case CL_SUCCESS:                            break;
		case CL_DEVICE_NOT_FOUND:                   printf("Error : Device not found. Line : %d\n",line); break;
		case CL_DEVICE_NOT_AVAILABLE:               printf("Error : Device not available. Line : %d\n",line); break;
		case CL_COMPILER_NOT_AVAILABLE:             printf("Error : Compiler not available. Line : %d\n",line); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:      printf("Error : Memory object allocation failure. Line : %d\n",line); break;
		case CL_OUT_OF_RESOURCES:                   printf("Error : Out of resources. Line : %d\n",line); break;
		case CL_OUT_OF_HOST_MEMORY:                 printf("Error : Out of host memory. Line : %d\n",line); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:       printf("Error : Profiling information not available. Line : %d\n",line); break;
		case CL_MEM_COPY_OVERLAP:                   printf("Error : Memory copy overlap. Line : %d\n",line); break;
		case CL_IMAGE_FORMAT_MISMATCH:              printf("Error : Image format mismatch. Line : %d\n",line); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:         printf("Error : Image format not supported. Line : %d\n",line); break;
		case CL_BUILD_PROGRAM_FAILURE:              printf("Error : Program build failure. Line : %d\n",line); break;
		case CL_MAP_FAILURE:                        printf("Error : Map failure. Line : %d\n",line); break;
		case CL_INVALID_VALUE:                      printf("Error : Invalid value. Line : %d\n",line); break;
		case CL_INVALID_DEVICE_TYPE:                printf("Error : Invalid device type. Line : %d\n",line); break;
		case CL_INVALID_PLATFORM:                   printf("Error : Invalid platform. Line : %d\n",line); break;
		case CL_INVALID_DEVICE:                     printf("Error : Invalid device. Line : %d\n",line); break;
		case CL_INVALID_CONTEXT:                    printf("Error : Invalid context. Line : %d\n",line); break;
		case CL_INVALID_QUEUE_PROPERTIES:           printf("Error : Invalid queue properties. Line : %d\n",line); break;
		case CL_INVALID_COMMAND_QUEUE:              printf("Error : Invalid command queue. Line : %d\n",line); break;
		case CL_INVALID_HOST_PTR:                   printf("Error : Invalid host pointer. Line : %d\n",line); break;
		case CL_INVALID_MEM_OBJECT:                 printf("Error : Invalid memory object. Line : %d\n",line); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    printf("Error : Invalid image format descriptor. Line : %d\n",line); break;
		case CL_INVALID_IMAGE_SIZE:                 printf("Error : Invalid image size. Line : %d\n",line); break;
		case CL_INVALID_SAMPLER:                    printf("Error : Invalid sampler. Line : %d\n",line); break;
		case CL_INVALID_BINARY:                     printf("Error : Invalid binary. Line : %d\n",line); break;
		case CL_INVALID_BUILD_OPTIONS:              printf("Error : Invalid build options. Line : %d\n",line); break;
		case CL_INVALID_PROGRAM:                    printf("Error : Invalid program. Line : %d\n",line); break;
		case CL_INVALID_PROGRAM_EXECUTABLE:         printf("Error : Invalid program executable. Line : %d\n",line); break;
		case CL_INVALID_KERNEL_NAME:                printf("Error : Invalid kernel name. Line : %d\n",line); break;
		case CL_INVALID_KERNEL_DEFINITION:          printf("Error : Invalid kernel definition. Line : %d\n",line); break;
		case CL_INVALID_KERNEL:                     printf("Error : Invalid kernel. Line : %d\n",line); break;
		case CL_INVALID_ARG_INDEX:                  printf("Error : Invalid argument index. Line : %d\n",line); break;
		case CL_INVALID_ARG_VALUE:                  printf("Error : Invalid argument value. Line : %d\n",line); break;
		case CL_INVALID_ARG_SIZE:                   printf("Error : Invalid argument size. Line : %d\n",line); break;
		case CL_INVALID_KERNEL_ARGS:                printf("Error : Invalid kernel arguments. Line : %d\n",line); break;
		case CL_INVALID_WORK_DIMENSION:             printf("Error : Invalid work dimension. Line : %d\n",line); break;
		case CL_INVALID_WORK_GROUP_SIZE:            printf("Error : Invalid work group size. Line : %d\n",line); break;
		case CL_INVALID_WORK_ITEM_SIZE:             printf("Error : Invalid work item size. Line : %d\n",line); break;
		case CL_INVALID_GLOBAL_OFFSET:              printf("Error : Invalid global offset. Line : %d\n",line); break;
		case CL_INVALID_EVENT_WAIT_LIST:            printf("Error : Invalid event wait list. Line : %d\n",line); break;
		case CL_INVALID_EVENT:                      printf("Error : Invalid event. Line : %d\n",line); break;
		case CL_INVALID_OPERATION:                  printf("Error : Invalid operation. Line : %d\n",line); break;
		case CL_INVALID_GL_OBJECT:                  printf("Error : Invalid OpenGL object. Line : %d\n",line); break;
		case CL_INVALID_BUFFER_SIZE:                printf("Error : Invalid buffer size. Line : %d\n",line); break;
		case CL_INVALID_MIP_LEVEL:                  printf("Error : Invalid mip-map level. Line : %d\n",line); break;
		default: printf("Error : Unknown. Line : %d\n",line); break;
	}

	if(err != CL_SUCCESS)
	{
		exit(-1);
	}

}

void fill_array(unsigned char *arr, int len)
{
	int i;
	
	for(i=0;i<len;i++)
	{
		arr[i] = (unsigned char)(i & 0xff);
	}
}

int main(int argc, char *argv[])
{
	int 			i, j, memSize;
	char 			platbuf[100], devname[100];
	double 			h2d, d2d, d2h;
	cl_int 			err;
	cl_uint 		numplatforms, numdevices, maxfreq, numunits;
	cl_ulong 		memsize;
	cl_platform_id 		*platforms;
	cl_device_id 		*devices;
	cl_event		event_write, event_copy, event_read;
	cl_command_queue 	cqCommandQue;
	cl_ulong 		start, end; 
	/*
	   Declare cl_mem buffer objects for the pinned host memory and the GPU 
	   device GMEM, respectively, and standard pointers to reference pinned host 
	   memory
	 */
	cl_context 		cxGPUContext; 
	cl_mem 			cmPinnedBufIn = NULL, cmPinnedBufOut = NULL, cmDevBufIn = NULL, cmDevBufOut = NULL; 
	unsigned char		*cDataIn = NULL; 
	unsigned char		*cDataOut = NULL; 

	if(argc==2)
	{	
		memSize = atoi(argv[1]);
	}
	else
	{
		printf("Syntax: ./pinned <transfer size>\n");
		exit(-1);
	}

	err = clGetPlatformIDs(0, 0, &numplatforms);
	if(err != CL_SUCCESS || numplatforms == 0)
	{
		printf("No platform found\n"); 
	}
	else
	{
		assert ((platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * numplatforms)) != NULL);		
		
		err = clGetPlatformIDs(numplatforms, platforms, NULL);
		check_opencl(err,__LINE__);

		for(i=0; i<numplatforms; i++)
		{
			err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platbuf),platbuf,NULL);
			check_opencl(err,__LINE__);
	
			printf("\nPlaform %s\n",platbuf);
			printf("------------------------\n");
			
			err = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 0, 0, &numdevices);
			if( err != CL_SUCCESS  || numdevices == 0)
			{
				printf("No devices found\n");
			}
			else
			{		
				assert ((devices = (cl_device_id *) malloc(sizeof(cl_device_id) * numdevices)) != NULL);		

				err = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, numdevices, devices, NULL);
				check_opencl(err,__LINE__);

				for(j=0; j<numdevices; j++)
				{
					err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(devname),devname,NULL);
					check_opencl(err,__LINE__);

					printf("\nDevice %s\n",devname);					

					cxGPUContext = clCreateContext(0,1,&devices[j],0,0,&err);	
					check_opencl(err,__LINE__);
				
					cqCommandQue = clCreateCommandQueue(cxGPUContext,devices[j],CL_QUEUE_PROFILING_ENABLE,&err);
					check_opencl(err,__LINE__);

					/*
					   Allocate cl_mem buffer objects for the pinned host memory and the GPU 
					   device GMEM, respectively
					 */
					cmPinnedBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, NULL); 
					cmPinnedBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memSize, NULL, NULL); 
					cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, memSize, NULL, NULL); 
					cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, memSize, NULL, NULL);

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
					err = clEnqueueWriteBuffer(cqCommandQue, cmDevBufIn, CL_FALSE, 0, memSize, cDataIn, 0, NULL, &event_write);
					check_opencl(err,__LINE__);
					clFinish(cqCommandQue);
					
					clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					h2d = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));

					/*
					   Copy data from GPU device GMEM to the GPU device GMEM
					 */
					err = clEnqueueCopyBuffer(cqCommandQue, cmDevBufIn, cmDevBufOut, 0, 0, memSize, 0, NULL, &event_copy);
					check_opencl(err,__LINE__);				
					clFinish(cqCommandQue);
					
					clGetEventProfilingInfo(event_copy, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_copy, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					d2d = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));

					/*
					   Read data from GPU device GMEM to pinned host memory
					 */
					err = clEnqueueReadBuffer(cqCommandQue, cmDevBufOut, CL_TRUE, 0, memSize, cDataOut, 0, NULL, &event_read);
					check_opencl(err,__LINE__);				
					clFinish(cqCommandQue);
						
					clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					d2h = ((double)memSize)/((end - start) * 1.0e-9 * (double)(1 << 20));
			
					/*
					   Check if data has been transferred successfully
					 */
					for(i=0; i<memSize; i++)
						assert(cDataOut[i] == cDataIn[i]);
				
					printf("Host to Device : %f MB/s\n",h2d);				
					printf("Device to Device : %f MB/s\n",d2d);				
					printf("Device to Host : %f MB/s\n",d2h);				

					clReleaseMemObject(cmPinnedBufIn);
					clReleaseMemObject(cmPinnedBufOut);
					clReleaseMemObject(cmDevBufIn);
					clReleaseMemObject(cmDevBufOut);

					clReleaseCommandQueue(cqCommandQue);
					clReleaseContext(cxGPUContext);
				}

				free(devices);	
			}	
				
			printf("------------------------\n");
		}
		
		free(platforms);	
	}	

	return 0;
}
