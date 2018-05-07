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

INDEX_OBJS=util.o index.o
SEARCH_OBJS=util.o search.o

util.o: util.cpp
	$(CXX) $(CXXFLAGS) -c -o util.o util.cpp $(LDFLAGS)

%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@ $(LDFLAGS)

index: $(INDEX_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(INDEX_OBJS) $(LDFLAGS)

search: $(SEARCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(SEARCH_OBJS) $(LDFLAGS)

evaluation: evaluation.cpp
	$(CXX) $(CXXFLAGS) -o $@ evaluation.cpp

prepare: prepare.sh
	./prepare.sh

clean:
	rm -rf *.o index search evaluation

veryclean: clean
	rm -rf gist_index_*_table sift_index_*_table gist_result_%_table sift_result_%_table

# begin build index commands and search index commands

buildgist2: index
	./index ./gist/gist_base.fvecs gist_index_2_table 2

buildgist4: index
	./index ./gist/gist_base.fvecs gist_index_4_table 4

buildgist8: index
	./index ./gist/gist_base.fvecs gist_index_8_table 8

buildgist16: index
	./index ./gist/gist_base.fvecs gist_index_16_table 16

searchgist2: search
	./search gist_index_2_table ./gist/gist_base.fvecs ./gist/gist_query.fvecs gist_result_2_table 2 50000 100 400

searchgist4: search
	./search gist_index_4_table ./gist/gist_base.fvecs ./gist/gist_query.fvecs gist_result_4_table 4 50000 100 400

searchgist8: search
	./search gist_index_8_table ./gist/gist_base.fvecs ./gist/gist_query.fvecs gist_result_8_table 8 50000 100 400

searchgist16: search
	./search gist_index_16_table ./gist/gist_base.fvecs ./gist/gist_query.fvecs gist_result_16_table 16 50000 100 400

buildsift2: index
	./index ./sift/sift_base.fvecs sift_index_2_table 2

buildsift4: index
	./index ./sift/sift_base.fvecs sift_index_4_table 4

buildsift8: index
	./index ./sift/sift_base.fvecs sift_index_8_table 8

buildsift16: index
	./index ./sift/sift_base.fvecs sift_index_16_table 16

searchsift2: search
	./search sift_index_2_table ./sift/sift_base.fvecs ./sift/sift_query.fvecs sift_result_2_table 2 50000 100 400

searchsift4: search
	./search sift_index_4_table ./sift/sift_base.fvecs ./sift/sift_query.fvecs sift_result_4_table 4 50000 100 400

searchsift8: search
	./search sift_index_8_table ./sift/sift_base.fvecs ./sift/sift_query.fvecs sift_result_8_table 8 50000 100 400

searchsift16: search
	./search sift_index_16_table ./sift/sift_base.fvecs ./sift/sift_query.fvecs sift_result_16_table 16 50000 100 400
