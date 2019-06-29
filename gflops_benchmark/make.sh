as -o func1.o func1.S
as -o func2.o func2.S
as -o func3.o func3.S
gcc -c main.c
gcc -o main main.o func2.o func1.o func3.o
