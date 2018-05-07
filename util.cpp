#include "util.h"

using namespace std;

void load_index(char* filename,  unsigned int *&codes, float *&matrix_projection, unsigned int &dim, unsigned int &ntable, unsigned int &npoint){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}

    //read projection matrix
    unsigned int codelen;

    in.read((char*)&dim,sizeof(int));
    in.read((char*)&codelen,sizeof(int));
    ntable = codelen / 32;
    matrix_projection = new float[codelen * dim];
    for (unsigned i = 0; i < codelen * dim; i++) {
        in.read((char*)&matrix_projection[i], sizeof(float));
    }

    // read codes
    in.read((char*)&npoint, sizeof(unsigned int));
    in.read((char*)&ntable, sizeof(unsigned int));
    codes = new unsigned int [npoint * ntable];
    for (size_t i = 0; i < ntable*npoint; i++) {
        in.read((char*)&(codes[i]),sizeof(unsigned int));
    }
    in.close();
    printf("Index loaded! dim %u, codelen %u ntable %u, npoints %u\n", dim, codelen, ntable, npoint);
}

void saveIndex(char* filename, unsigned int *codes, float *matrix_projection, unsigned int dim, unsigned int ntables, unsigned int npoints){
    std::ofstream out(filename,std::ios::binary);
    if(!out.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned int codelen = ntables * 32;

    out.write((char*)&dim, sizeof(unsigned int));
    out.write((char*)&codelen, sizeof(unsigned int));

    //write projection matrix
    for (unsigned i = 0; i < codelen * dim; i++) {
        out.write((char*)&matrix_projection[i], sizeof(float));
    }

    //write codes
    out.write((char*)&npoints, sizeof(unsigned int));
    out.write((char*)&ntables, sizeof(unsigned int));
    for (size_t i = 0; i < ntables*npoints; i++) {
        out.write((char*)&codes[i], sizeof(unsigned int));
    }
    printf("index saved! dim %u, ntable %u, npoint %u\n", dim, ntables, npoints);
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

void saveKNNResults(char* filename, unsigned int *result, int nquery, int K){
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

void init_matrix_size(sMatrixSize &matrix_size, size_t npoints, int dim, int codelen){
    matrix_size.uiHA = npoints;
    matrix_size.uiWA = dim;

    matrix_size.uiHB = dim;
    matrix_size.uiWB = codelen;

    matrix_size.uiHC = npoints;
    matrix_size.uiWC = codelen;

}