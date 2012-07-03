NVCC = /usr/local/cuda/bin/nvcc
CUDAPATH = /usr/local/cuda

MPIPATH = /usr/local/mpich2-1.0.7

NVCCFLAGS = -I$(CUDAPATH)/include -arch=sm_11 
INCLUDE = -I$(MPIPATH)/include
LIBS = -L$(CUDAPATH)/lib64 -L/$(MPIPATH)/lib 
FLAGS = -DMPICH_IGNORE_CXX_SEEK -lcuda -lcudart -lm -lcublas -lmpich

RM = rm -rf

FILES = mpi_bandwidth all_mpi_bandwidth new_mpi_bandwidth dgemm_mpi dgemm_mpi_new

all : $(FILES)

$(FILES) : %:%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(FLAGS) $< -o $@

clean :
	$(RM) $(FILES)
