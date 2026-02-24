OLD  := MMult_cuBLAS_1
NEW := MMult_cuda_bf16_5
SMS ?= 80 86 90

#
# sample makefile for bf16 input
#

CUDA_PATH  ?= /usr/local/cuda
OPENBLAS_PATH ?= /home/khj/miniconda3/pkgs/openblas-0.3.31-pthreads_h6ec200e_0
CC         := nvcc 
LINKER     := $(CC)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
CFLAGS     := -std=c++17 -O2 -I$(CUDA_PATH)/include -I$(OPENBLAS_PATH)/include
LDFLAGS    := -L$(CUDA_PATH)/lib64 -L$(OPENBLAS_PATH)/lib -lm -lcublas -lopenblas

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST_OBJS  := test_MMult_bf16.o $(NEW).o 

%.o: %.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all: 
	make clean;
	make test_MMult_bf16.x

test_MMult_bf16.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult_bf16.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
