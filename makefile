CC = g++
CFLAGS = -Wall -fopenmp -O3 -march=native -ffast-math -std=c++11

all: PureSlim.o main.cpp
	$(CC) $(CFLAGS) -o slim main.cpp PureSlim.o

PureSlim.o: PureSlim.hpp PureSlim.cpp
	$(CC) $(CFLAGS) -c PureSlim.cpp -o $@
