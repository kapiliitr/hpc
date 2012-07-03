CC = g++

CLINC = -I/usr/local/cuda/include
CLINKLIB = -L/usr/local/cuda/lib64
CLLINKFLAGS = -lOpenCL

RM = rm -rf

CFILES = demo array pinned matmatmult 
CPPFILES = 

all : $(CFILES) $(CPPFILES)

$(CFILES) : %:%.c
	$(CC) $(CLLINKFLAGS) $(CLINC) $(CLINKLIB) $< -o $@

$(CPPFILES) : %:%.cpp
	$(CC) $(CLLINKFLAGS) $(CLINC) $(CLINKLIB) $< -o $@
  
clean :
	$(RM) $(CFILES) $(CPPFILES)
