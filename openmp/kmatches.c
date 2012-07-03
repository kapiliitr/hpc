#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<sys/time.h>

int main(int argc, char **argv)
{
	int 		*array, numthreads, size, data, i, parCount, serCount;
	struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead, ser_overhead;
	
	if(argc!=4)
	{
		printf("Syntax : exec <threads> <size> <number>\n");
		exit(1);
	}

	numthreads = atoi(argv[1]);
	size = atoi(argv[2]);
	data = atoi(argv[3]);
	
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
	
	parCount = 0;

        gettimeofday(&TimeValue_Start, &TimeZone_Start);

	omp_set_num_threads(numthreads);

	#pragma omp parallel for default(none) private(i) shared(array, size, parCount, data)
	for(i=0;i<size;i++)
	{
		#pragma omp critical
		if(array[i]==data)
		{
			parCount++;
		}		
	}
		
	gettimeofday(&TimeValue_Final, &TimeZone_Final);

        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start)/1000000.0;
	
        gettimeofday(&TimeValue_Start, &TimeZone_Start);
	
	serCount = 0;
	for(i=1;i<size;i++)
	{
		if(array[i]==data)
			serCount++;
	}

	gettimeofday(&TimeValue_Final, &TimeZone_Final);

        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        ser_overhead = (time_end - time_start)/1000000.0;

	printf("Count Parallel : %d Time Overhead : %f\n",parCount,time_overhead);
	printf("Count Serial : %d Time Overhead : %f\n",serCount,ser_overhead);
	
	free(array);
	
	return 0;
}


