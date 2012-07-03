#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>

#define NUM 5

/*void *func(void *param)
{
	printf("Hello %i ! I am process %i.\n",pthread_self(),(int)param);
	pthread_exit(NULL);
}*/

void *func(int *param)
{
	printf("Hello %i ! I am process %i.\n",pthread_self(),param);
	pthread_exit(NULL);
}

int main()
{
	pthread_t thread[NUM];
	int err; 
	int i;
	for(i=0;i<NUM;i++)
	{
		printf("Creating thread %d.\n",i);
		if(err=pthread_create(&thread[i], NULL, func, (void *)i))
		{	
			printf("Error %d creating thread %d.\n",err,i);
			exit(EXIT_FAILURE);
		}
	}
	pthread_exit(NULL);	
	return 0;
}
