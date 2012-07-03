#
# Makefile for openmp programs
#
# Author : Kapil Agarwal
#
# Date : 26 May 2012
#

FILES = findmin kmatches

CC = gcc
CFLAGS = -fopenmp
LDFLAGS = -lm
RM = rm -rf

all: $(FILES)

$(FILES) : %:%.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) 

clean:
	$(RM) *.o $(FILES)
