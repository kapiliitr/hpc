#
# Makefile for MPI programs
#
# Author : Kapil Agarwal
#
# Date : 28 May 2012
#

# list of files 
FILES = 1 3.3 3.7 infnorm mpi fileWrite transpose

CC = mpicc			# name of compiler 
CFLAGS = 			# warning flag
LDFLAGS = -lm			# linker flags
RM = rm -rf			# remove files

all : $(FILES)

$(FILES):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS)

clean : 
	$(RM) *.o *.out $(FILES)
