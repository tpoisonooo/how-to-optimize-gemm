OLD  := MMult_4x4_19
NEW  := MMult_4x4_19

#
# sample makefile
#

CC         := g++
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
CFLAGS     := -std=c++11 -O3 -march=armv7-a -mfpu=neon -ftree-vectorize
LDFLAGS    := -lm

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST_OBJS  := test_MMult.o $(NEW).o 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
