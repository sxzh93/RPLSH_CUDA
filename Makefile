.PHONY: clean prepare buildgist* searchgist* buildsift* searchsift*

EXECUTABLE=index search evaluation

default: $(EXECUTABLE)

LDFLAGS=-I/usr/local/cuda-9.0/samples/common/inc -L/usr/local/cuda-9.0/lib64/ -lcudart -lcurand -lcublas

NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 -m64 --gpu-architecture compute_61

CXX=g++
CXXFLAGS=-std=c++11 -m64 -O3 -Wall -g

CC=gcc
CFLAGS=-std=c++11 -O3 -m64 -Wall


util.o: util.cpp
	$(CXX) $(CXXFLAGS) -c -o util.o util.cpp $(LDFLAGS)

%.o: %.cu 
	$(NVCC) $< $(NVCCFLAGS) -c -o $@ $(LDFLAGS)

index: index.o util.o 
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

search: search.o util.o 
	$(CXX) $(CXXFLAGS) -o $@ search.o $(LDFLAGS)

evaluation: evaluation.cpp
	$(CXX) $(CXXFLAGS) -o $@ evaluation.cpp

prepare: prepare.sh
	./prepare.sh

buildgist%: index
	./index ./gist/gist_base.fvecs gist_index_%_table %

searchgist%: search
	./search gist_index_%_table ./gist/gist_base.fvecs ./gist/gist_query.fvecs gist_result_%_table % 50000 100 400

buildsift%: index
	./index ./sift/sift_base.fvecs sift_index_%_table %

searchsift%: search
	./search sift_index_%_table ./sift/sift_base.fvecs ./sift/sift_query.fvecs sift_result_%_table % 50000 100 400

clean:
	rm -rf *.o index search evaluation
