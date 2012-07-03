#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matmatmult(__global double *vecA, __global double *vecB, __global double *vecC, int dim)
{
   int tx, ty, k;
   double value = 0.0;

   tx = get_global_id(0); 
   ty = get_global_id(1);
 
   for (k = 0; k < dim; k++)
   {
      value += vecA[tx * dim + k] * vecB[k * dim + ty];
   }
 
   vecC[tx * dim + ty] = value;
}
