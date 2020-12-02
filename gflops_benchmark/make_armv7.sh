as -o func1.o -mfpu=neon func1_armv7.S
as -o func2.o -mfpu=neon func2_armv7.S
gcc -c -O3  main.c
gcc -o main main.o func2.o func1.o
