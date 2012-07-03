#
# Makefile for pthread programs
#
# Author : Kapil Agarwal
#
# Date : 25 May 2012
#

# list of files 
FILES = create join stack condvar findmin 

CC = gcc			# name of compiler 
CFLAGS = -w			# warning flag
LDFLAGS = -lpthread -lm		# linker flags
RM = rm -rf			# remove files

all : $(FILES)

$(FILES):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS)

clean : 
	$(RM) *.o $(FILES)
