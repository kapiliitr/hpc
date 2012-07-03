#include<stdio.h>

int main()
{
	int dev;
	cudaDeviceProp devprop;
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&devprop,dev);
	printf("name = %s\ntotal global mem = %1fM\nshared mem per block = %1fK\nregs per block = %d\nwarp size = %d\nclock rate = %1fGHz\nmax threads per block= %d\ntotal const mem = %1fK\nmultiprocessor count = %d\nmax threads per multiprocessor = %d\nl2 cache size = %1fK\n",devprop.name,devprop.totalGlobalMem/(1024*1024.0),devprop.sharedMemPerBlock/1024.0,devprop.regsPerBlock,devprop.warpSize,devprop.clockRate/(1000000.0),devprop.maxThreadsPerBlock,devprop.totalConstMem/1024.0,devprop.multiProcessorCount,devprop.maxThreadsPerMultiProcessor,devprop.l2CacheSize/1024.0);
}
