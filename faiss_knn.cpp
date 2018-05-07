#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <iostream>
#include <fstream>

using namespace std;

void saveResults(char* filename, long *result, int nquery, int K){
    std::ofstream out(filename,std::ios::binary);
    for(int i=0; i<nquery; i++){
        out.write((char*)&K, sizeof(int));
        for(int j=0; j<K; j++){
            int id = result[i*K + j];
            out.write((char*)&id, sizeof(int));
        }
    }
    out.close();
}

void load_data(char* filename, float*& data, unsigned int& num, unsigned int& dim) { // load data with sift10K pattern
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {cout << "open file error" << endl; exit(-1);}
    //read dim
    in.read((char*)&dim, 4);

    //read fsize
    in.seekg(0, ios::end);
    ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;

    //read number of vector(dim + data)
    num = fsize / (dim + 1) / 4;

    //read data
    data = new float[num * dim];
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
    cout << "load data okay, npoints " << num << ", dim " << dim << endl;
}

int main(int argc, char **argv) {
    unsigned int base_dim = 0;
    unsigned int nbase = 0;
    unsigned int query_dim = 0;
    unsigned int nquery = 0;
    float *base_matrix = NULL;
    float *query_matrix = NULL;

    if (argc != 6) {
        cout<< argv[0] << " <base_file> <query_file> <result_file> <nlist> <K>";
    }

    char *base_file = argv[1];
    char *query_file = argv[2];
    char *result_file = argv[3];
    int nlist = atoi(argv[4]);
    int K = atoi(argv[5]);

    // load data and query
    load_data(base_file, base_matrix, nbase, base_dim);
    load_data(query_file, query_matrix, nquery, query_dim);

    faiss::gpu::StandardGpuResources res;

    assert(base_dim == query_dim);

    int nbits = 8;          // with 8 bits per subquantizer
    int m = base_dim / nbits;
    faiss::gpu::GpuIndexIVFPQ index_pq(&res, base_dim, nlist, m, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search

    index.train(nbase, base_matrix);
    index.add(nbase, base_matrix);

    long *I = new long[K*nquery];
    float *D = new float[K*nquery];

    auto s = chrono::high_resolution_clock::now();
    index_pq.search(nquery, query_matrix, K, D, I);
    auto e = chrono::high_resolution_clock::now();

    // report time
    chrono::duration<double> diff = e-s;
    cout<<"faiss query searching time: "<<diff.count()<<endl;

    saveResults(result_file, I, nquery, K);

    delete [] I;
    delete [] D;
    delete [] query_matrix;
    delete [] base_matrix;

    return 0;
}
