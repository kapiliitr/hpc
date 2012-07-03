#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cublas_v2.h"

void safe_call(cublasStatus_t ret, int line)
{
	if(ret!=CUBLAS_STATUS_SUCCESS)
	{
		printf("Error at line %d : %s\n",line);
		exit(-1);
	}
}

int main()
{
	cublasHandle_t handle;
	int version;
	
	safe_call(cublasCreate(&handle),__LINE__);
	
	safe_call(cublasGetVersion(handle,&version),__LINE__);

	printf("CUBLAS version = %d\n",version);

	safe_call(cublasDestroy(handle),__LINE__);
}
