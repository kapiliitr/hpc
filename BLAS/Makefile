include ./Make.inc

############################
# Object files
############################

OBJECTS=mat_mat_mult


##############################

$(OBJECTS): $(OBJECTS).o
	$(CC) -o $(OBJECTS) $(OBJECTS).o -L$(CBLAS_LIB) $(CBLAS_FLAGS) 
	mv $(OBJECTS) $(BIN)  	

$(OBJECTS).o: $(OBJECTS).c
	$(CC) -c $(CCFLAGS) -DMKL_ILP64 -I$(CBLAS_INCLUDE)  $(OBJECTS).c


clean:
	rm -f *.o $(OBJECTS)
	rm -f  $(BIN)/$(OBJECTS)

