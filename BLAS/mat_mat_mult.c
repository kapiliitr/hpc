#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>

#include<mkl.h>

#define loop(i,n) for(i=0;i<n;i++)
#define max atoi(argv[1])

int main(int argc, char ** argv)
{
//  int s[max][max];
//  int m[max][max];
//  int n[max][max];
    double ** s, ** m, ** n;
    double *x, *y, *z;
    s = (double **) malloc(max*sizeof(double *)); 	
    m = (double **) malloc(max*sizeof(double *)); 	
    n = (double **) malloc(max*sizeof(double *)); 	 
    if(s==0 || m==0 || n==0) { printf("Failed\n");  }
    int i,j,k,t;
    loop(i,max)
    {
	s[i] = (double *) malloc(max*sizeof(double));
	m[i] = (double *) malloc(max*sizeof(double));
	n[i] = (double *) malloc(max*sizeof(double));
	if(s[i]==0 || m[i]==0 || n[i]==0) { printf("Failed\n");  }
    }
    x = (double *) malloc(max*max*sizeof(double));
    y = (double *) malloc(max*max*sizeof(double));
    z = (double *) malloc(max*max*sizeof(double));
    struct timeval time1,time2;
    long long timetaken;
    double mflops;
/*    double s[max][max];
    double m[max][max];
    double n[max][max];
*/
    //input matrix1
    srand(time(NULL));
    loop(i,max)
    {
        loop(j,max)
        {
            //scanf("%d",&m[i][j]);
	    m[i][j] = rand()%100;	
	    x[i*max+j] = m[i][j];
        }
    }

    //input matrix2
    srand(time(NULL));
    loop(i,max)
    {
        loop(j,max)
        {
            //scanf("%d",&n[i][j]);
	    n[i][j] = rand()%100;	
	    y[i*max+j] = n[i][j];
        }
    }
	
    gettimeofday(&time1,NULL);
    //perform multiplication
    /*loop(i,max)
    {
        loop(j,max)
        {
            //s[i][j]=0;
	    double v=0.0;	
            //double v=0.0;		
            loop(k,max)
            {
                v+=(m[i][k]*n[k][j]);
            }
	    s[i][j]=v;
        }
    }*/
    cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, max, max , max, 1.0, x , max , y , max, 0.0, z, max);
    gettimeofday(&time2,NULL);
   
    timetaken =  (long long) (time2.tv_sec - time1.tv_sec)*1000000 + (time2.tv_usec - time1.tv_usec);	
    printf("TIME = %lld\n", timetaken);

    mflops = (2.0*max*max*max)/(1000*timetaken);
    printf("GFLOPS = %lf \n",mflops);    	    

    //output product
/*    loop(i,max)
    {
        loop(j,max)
        {
            printf("%d ",s[i][j]);
        }
        printf("\n");
    }
*/
    loop(i,max)
    {
	free(s[i]);	
	free(m[i]);	
	free(n[i]);	
    }

    free(s);
    free(m);
    free(n);
    
    return 0;
}
