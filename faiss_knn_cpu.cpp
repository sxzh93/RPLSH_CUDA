#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <iostream>
#include <fstream>

using namespace std;

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
    //char *result_file = argv[3];
    int nlist = atoi(argv[4]);
    int K = atoi(argv[5]);

    // load data and query
    load_data(base_file, base_matrix, nbase, base_dim);
    load_data(query_file, query_matrix, nquery, query_dim);

    faiss::IndexFlatL2 quantizer(base_dim);       // the other index
    faiss::IndexIVFFlat index(&quantizer, base_dim, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search
    assert(!index.is_trained);
    index.train(nbase, base_matrix);
    assert(index.is_trained);
    index.add(nbase, base_matrix);

    // begin search
    long *I = new long[K*nquery];
    float *D = new float[K*nquery];

    auto s = chrono::high_resolution_clock::now();
    index.search(nquery, query_matrix, K, D, I);
    auto e = chrono::high_resolution_clock::now();

    // report time
    chrono::duration<double> diff = e-s;
    cout<<"faiss query searching time: "<<diff.count()<<endl;

    delete [] I;
    delete [] D;
    delete [] query_matrix;
    delete [] base_matrix;

    return 0;

}
