.PHONY: clean prepare

EXECUTABLE=index search

default: $(EXECUTABLE)

LDFLAGS=-I/usr/local/cuda-9.0/samples/common/inc -L/usr/local/cuda-9.0/lib64/ -lcudart -lcurand -lcublas -lm

CC=gcc
CFILES=cycletimer.c
HFILES=cycletimer.h
CFLAGS=-g -O3 -Wall

NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 -m64 --gpu-architecture compute_61

CXX=g++
CXXFLAGS=-std=c++11 -m64 -O3 -Wall -g

cycletimer.o: cycletimer.c cycletimer.h
	$(CC) $(CFLAGS) -o $@ $(CFILES) -lm

%.o: %.cu cycletimer.o
	$(NVCC) $< $(NVCCFLAGS) -c -o $@ $(LDFLAGS)

index: index.o
	$(CXX) $(CXXFLAGS) -o $@ index.o $(LDFLAGS)

search: search.o cycletimer.o
	$(CXX) $(CXXFLAGS) -o $@ search.o $(LDFLAGS)

prepare: prepare.sh
	./prepare.sh

clean:
	rm -rf *.o index search
