as -o func1.o -march=armv7-a -mfloat-abi=softfp -mfpu=neon func1_armv7.S
as -o func2.o -march=armv7-a -mfloat-abi=softfp -mfpu=neon func2_armv7.S
gcc -c main.c
gcc -o main main.o func2.o func1.o
