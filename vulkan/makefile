OLD  := MMult_vulkan_1
NEW := MMult_vulkan_1

#
# sample makefile
#

CC         := g++ 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
CFLAGS     := -std=c++17 -O2 -g
LDFLAGS    := -lm -lopenblas

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST_OBJS  := test_MMult.o $(NEW).o 

%.o: %.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
