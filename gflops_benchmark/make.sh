rm -rf main
g++ -c func1.S
g++ -c func2.S
g++ -c main.cpp
g++ -o main main.o func1.o func2.o
rm *.o
