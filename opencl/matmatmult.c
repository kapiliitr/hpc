#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#include<string.h>
#include<CL/cl.h>

#define ERROR 1.0e-12

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

char* read_source(char *source_path)
{
	FILE 	* pFile;
	char 	*source_string;
	size_t 	len;

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

void print(double *arr, int len)
{
	printf("----------------------------------\n");
	for(int i=0; i<len; i++)
	{
		for(int j=0; j<len; j++)
		{
			printf("%5f   ",arr[i*len+j]);
		}
		printf("\n");
	}
	printf("----------------------------------\n");
}

int main(int argc, char *argv[])
{
	int 			i, j, SIZE;
	char 			platbuf[100], devname[100];
	double 			*h_A, *h_B, *h_C, gflops;
	cl_int 			err;
	cl_uint 		numplatforms, numdevices, maxfreq, numunits;
	cl_ulong 		memsize, start, end; 
	cl_platform_id 		*platforms;
	cl_device_id 		*devices;
	cl_context		context;
	cl_command_queue 	cmdqueue;
	cl_program		program;
	cl_kernel		kernel;
	cl_event		event;
	cl_mem			d_A, d_B, d_C ;	
	size_t			globalWorkSize[2], localWorkSize[2];

	if(argc==3)
	{	
		SIZE = atoi(argv[1]);
		if(atoi(argv[2]) > 32)
		{
			printf("Maximum work group size is 32\n");
			exit(-1);
		}
		if(SIZE%atoi(argv[2]) != 0)
		{
			printf("Work group size should be multiple of Matrix dimension\n");
			exit(-1);
		}
	}
	else
	{
		printf("Syntax: ./matmatmult <matrix size> <work group size>\n");
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
	
			printf("\nPlaform : %s\n",platbuf);
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

					printf("\nDevice : %s\n",devname);				

					context = clCreateContext(0,1,&devices[j],0,0,&err);	
					check_opencl(err,__LINE__);
					
					char * kernel_source = read_source("matmatmult_kernel.cl");
					size_t kernel_size = strlen(kernel_source);
					
					cmdqueue = clCreateCommandQueue(context,devices[j],CL_QUEUE_PROFILING_ENABLE,&err);
					check_opencl(err,__LINE__);

					program = clCreateProgramWithSource(context,1,(const char **) &kernel_source,&kernel_size,&err);
					check_opencl(err,__LINE__);

					err = clBuildProgram(program,1,&devices[j],NULL,NULL,NULL);
					if (err != CL_SUCCESS)
					{
						char buildLog[16384];
						clGetProgramBuildInfo(program, devices[j], CL_PROGRAM_BUILD_LOG,sizeof(buildLog), buildLog, NULL);
						printf("Error in kernel : \n%s\n",buildLog);
						clReleaseProgram(program);
						exit(-1);
					}
					
					kernel = clCreateKernel(program,"matmatmult",&err);
					check_opencl(err,__LINE__);

				        d_A = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(double)*SIZE*SIZE,h_A,&err);
					check_opencl(err,__LINE__);
			
				        d_B = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(double)*SIZE*SIZE,h_B,&err);
					check_opencl(err,__LINE__);
	        
					d_C = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(double)*SIZE*SIZE,NULL,&err);
					check_opencl(err,__LINE__);

					err = clSetKernelArg(kernel,0,sizeof(cl_mem),&d_A);
					check_opencl(err,__LINE__);

					err = clSetKernelArg(kernel,1,sizeof(cl_mem),&d_B);
					check_opencl(err,__LINE__);

					err = clSetKernelArg(kernel,2,sizeof(cl_mem),&d_C);
					check_opencl(err,__LINE__);

					err = clSetKernelArg(kernel,3,sizeof(int),(void *)&SIZE);
					check_opencl(err,__LINE__);
			
					err = clEnqueueNDRangeKernel(cmdqueue,kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,&event);
					check_opencl(err,__LINE__);
				
					err = clFinish(cmdqueue);
					check_opencl(err,__LINE__);
					
					clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL); 
					clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
					/*
					   clGetEventProfilingInfo() returns time in nanoseconds
					 */
					gflops = (( 2.0 * SIZE * SIZE * SIZE )/(end - start));

					err = clEnqueueReadBuffer(cmdqueue,d_C,CL_TRUE,0,sizeof(double)*SIZE*SIZE,h_C,0,NULL,NULL);
					check_opencl(err,__LINE__);

					if(check(h_A,h_B,h_C,SIZE))
						printf("Result verification success\nGflops : %f\n",gflops);
					else
						printf("Result verification failed\n");
					
					/*print(h_A,SIZE);
					print(h_B,SIZE);
					print(h_C,SIZE);
					*/

					clReleaseMemObject(d_A);
					clReleaseMemObject(d_B);
					clReleaseMemObject(d_C);
					clReleaseProgram(program);
					clReleaseKernel(kernel);
					clReleaseCommandQueue(cmdqueue);
					clReleaseContext(context);
				}

				free(devices);	
			}	
				
			printf("------------------------\n");
		}
		
		free(platforms);	
	}	

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
