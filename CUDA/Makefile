NVCC = /usr/local/cuda/bin/nvcc
CUDAPATH = /usr/local/cuda

NVCCFLAGS = -I$(CUDAPATH)/include -arch=sm_11
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm -lcublas

RM = rm -rf

FILES = devprop vecadd matmatmult cublas bandwidth cublas_squarematmatmult bandwidth_range NaN/cublas_matmatmult NaN/cdac_cublas_mat_mat_mult bandwidth_pageable

all : $(FILES)

$(FILES) : %:%.cu
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) $< -o $@

clean :
	$(RM) $(FILES)
