#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>
#include<assert.h>

void check_opencl(cl_int err, int line)
{
	if(err != CL_SUCCESS)
	{
		printf("Error:%d Line:%d\n",err,line);
		exit(-1);
	}
}

int main()
{
	int 		i, j;
	char 		platbuf[100], devname[100];
	cl_int 		err;
	cl_uint 	numplatforms, numdevices, maxfreq, numunits;
	cl_ulong 	memsize;
	cl_platform_id 	*platforms;
	cl_device_id 	*devices;
	
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
					printf("\t---------------------------Device details-------------------------------------\n");
					printf("\tPlatform Name                          :  %s\n",platbuf);

					err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(devname),devname,NULL);
					check_opencl(err,__LINE__);
					printf("\tDevice Name                            :  %s\n",devname);

					err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numunits),&numunits,NULL);
					check_opencl(err,__LINE__);
					printf("\tNumber of compute units                :  %u\n",numunits);

					err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxfreq),&maxfreq,NULL);
					check_opencl(err,__LINE__);
					printf("\tClock Frequency                        :  %f GHz\n",maxfreq/1000.0);

					err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memsize),&memsize,NULL);
					check_opencl(err,__LINE__);
					printf("\tMemory Size                            :  %f GB\n",memsize/(1024*1024*1024.0));
				}
				
				free(devices);	
			}	
		}
		
		free(platforms);	
	}

	return 0;
}
