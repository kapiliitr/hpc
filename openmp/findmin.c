#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<sys/time.h>

#define MIN 2147483647

int main(int argc, char **argv)
{
	int 		*array, numthreads, size, i, myMin, serMin;
	struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead, ser_overhead;
	
	if(argc!=3)
	{
		printf("Syntax : exec <threads> <size>\n");
		exit(1);
	}

	numthreads = atoi(argv[1]);
	size = atoi(argv[2]);
	
	if(numthreads<1)
	{
		printf("There should be atleast 1 thread.\n");
		exit(1);
	}

	if(size<1)
	{
		printf("There should be atleast 1 element.\n");
		exit(1);
	}

	array = (int *)malloc(size*sizeof(int));
	
	srand48(time(NULL));
	for(i=0;i<size;i++)
	{
		array[i] = mrand48();
	}
	
	myMin = MIN;

        gettimeofday(&TimeValue_Start, &TimeZone_Start);

	omp_set_num_threads(numthreads);
	
	#pragma omp parallel for default(none) private(i) shared(array, size, myMin) schedule(dynamic)
	for(i=0;i<size;i++)
	{
		#pragma omp critical
		if(array[i]<myMin)
		{
			myMin = array[i];
		}		
	}
		
	gettimeofday(&TimeValue_Final, &TimeZone_Final);

        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start)/1000000.0;
	
        gettimeofday(&TimeValue_Start, &TimeZone_Start);
	
	serMin = array[0];
	for(i=1;i<size;i++)
	{
		if(array[i]<serMin)
			serMin = array[i];
	}

	gettimeofday(&TimeValue_Final, &TimeZone_Final);

        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        ser_overhead = (time_end - time_start)/1000000.0;

	printf("Min Parallel : %d Time Overhead : %f\n",myMin,time_overhead);
	printf("Min Serial : %d Time Overhead : %f\n",serMin,ser_overhead);
	
	free(array);
	
	return 0;
}


