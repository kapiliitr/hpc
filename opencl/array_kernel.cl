#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vectvectadd(__global double *vecA, __global double *vecB, __global double *vecC)
{
	size_t id = get_global_id(0);

	vecC[id] = vecA[id] + vecB[id];
}
