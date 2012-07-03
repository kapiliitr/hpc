#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<sys/time.h>

#define MAXSIZE atoi(argv[2])
#define MAXTHREADS 8
#define MAXNUM 10000
#define INFINITY (MAXNUM + 1)

struct data
{
	int *fullarr;
	int fullsize;
	int partsize;
};

int min = INFINITY;
pthread_mutex_t mutex;
struct data dataitem;

void *findmin(int *arg)
{
	int i, j, start, end, len, mymin;	
	int *array;
	i = arg;

	len = dataitem.partsize;
	start = i*len;
	end = start + len;
	array = dataitem.fullarr;

	mymin = array[start];
	for(j=start+1;j<end;j++)
	{
		if(array[j]<mymin)
		{
			mymin = array[j];
		}
	}

	pthread_mutex_lock (&mutex);
	if(min>mymin){
		min = mymin;
	}
	pthread_mutex_unlock (&mutex);

	pthread_exit(NULL);	
}

int main(int argc, char *argv[])
{
// Inititalisation
	pthread_t *threads;
	int i, j, numthreads, partialsize, min_nopthread;
	int *arr;
	pthread_attr_t attr;
	void *status;
	double time_start, time_end, diff;
        struct timeval tv;
        struct timezone tz;
		
// Check for number of arguments
	if(argc!=3)
	{
		printf("Enter number of threads and size of array as arguments.\n");
		exit(1);	
	}
	
	numthreads = atoi(argv[1]);
// Check for maximum number of threads
	if(numthreads>MAXTHREADS)
	{
		printf("Maximum number of threads is 8.\n");
		exit(1);	
	}

// Check for number of threads is factor of MAXSIZE
	if(MAXSIZE%numthreads!=0)
	{
		printf("Number of threads should be factor of %d.\n",MAXSIZE);
		exit(1);	
	}

// Allocate memory to threads
	threads  = (pthread_t *)malloc(numthreads*sizeof(pthread_t));

// Initialise mutex
	pthread_mutex_init(&mutex, NULL);

// Create threads in joinable state	
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

// Fill array
	arr = (int *)malloc(MAXSIZE*sizeof(int));
	srand(time(NULL));
	for(i=0;i<MAXSIZE;i++)
		arr[i] = rand()%MAXNUM + 1;

// Display the array
/*
	for(i=0;i<MAXSIZE;i++)
		printf("%d ",arr[i]);
	printf("\n");
*/

// Create threads and do the work
	partialsize = MAXSIZE/numthreads;
	dataitem.fullsize = MAXSIZE;
	dataitem.fullarr = arr;
	dataitem.partsize = partialsize;

// Calculating start time 
        gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	for(i=0;i<numthreads;i++)
	{
		pthread_create(&threads[i], &attr, findmin, (void *)i);
	}
	
	pthread_attr_destroy(&attr);

// Wait on the other threads
	for(i=0; i<numthreads; i++)
        {
		pthread_join(threads[i], &status);
	}

// Calculating end time
        gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	
	printf ("Min =  %d\n", min);
	printf("Time in Seconds (T)  :  %lf\n", time_end - time_start);

// CHECK WITHOUT PTHREADS
	min_nopthread = INFINITY;
        gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	for(i=0;i<MAXSIZE;i++)
	{
		if(min_nopthread>arr[i])
			min_nopthread = arr[i];
	}
        gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	printf("Min_nopthread = %d\n",min_nopthread);
	printf("Time in Seconds (T)  :  %lf\n", time_end - time_start);

// Clean and exit
	free(arr);
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
	
	return 0;
}

