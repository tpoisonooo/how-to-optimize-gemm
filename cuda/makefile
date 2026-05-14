OLD  := MMult_cuBLAS_1
#NEW  := MMult_cuBLAS_1
NEW := MMult_cuda_11
SMS ?= 70 75 80 86 90

#
# sample makefile
#

CUDA_PATH  ?= /usr/local/cuda
OPENBLAS_PATH ?= /usr
CC         := nvcc
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# CFLAGS     := -std=c++17 -O0 -g -G
CFLAGS     := -std=c++17 -O2 -I$(CUDA_PATH)/include -I$(OPENBLAS_PATH)/include/x86_64-linux-gnu -I$(OPENBLAS_PATH)/include/x86_64-linux-gnu/openblas-pthread
LDFLAGS    := -L$(CUDA_PATH)/lib64 -L$(OPENBLAS_PATH)/lib/x86_64-linux-gnu -L$(OPENBLAS_PATH)/lib/x86_64-linux-gnu/openblas-pthread -lm -lcublas -lopenblas

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
