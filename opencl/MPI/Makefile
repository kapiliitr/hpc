CC = mpicxx

CUDAPATH = /usr/local/cuda
MPIPATH = /usr/local/mpich2-1.0.7

INCLUDE = -I$(CUDAPATH)/include -I$(MPIPATH)/include
LIBS = -L$(CUDAPATH)/lib64 -L/$(MPIPATH)/lib 
FLAGS = -DMPICH_IGNORE_CXX_SEEK -lm -lcublas -lmpich -lOpenCL

RM = rm -rf

FILES = mpi_matmatmult_naive new_mpi_matmatmult mpi_bandwidth_pinned 

all : $(FILES)

$(FILES) : %:%.c
	$(CC) $(INCLUDE) $(LIBS) $(FLAGS) $< -o $@

clean :
	$(RM) $(FILES)
