#include <iostream>
#include <fstream>
#include <chrono>

#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h> // generate normal distribution



#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

using namespace std;

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;


void load_data(char* filename, float*& data, size_t& num, int& dim) { // load data with sift10K pattern
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

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// vertion 1, using if braching
void binarize_cpu_v1(float* result_matrix, unsigned int* codes, int npoints, int codelen){
    int table_id;
    int offset;
    unsigned int bit;
    for(int i=0; i<npoints; i++){
        for(int j=0; j<codelen; j++){
            offset = i*codelen + j;
            if(result_matrix[offset] > 0){
                bit = 1u << (j % 32);
                codes[table_id*npoints + i] |= bit;
            }
        }
    }
}


// vertion 2, binarize without if
void binarize_cpu_v2(float* result_matrix, unsigned int* codes, int npoints, int codelen){
    int table_id;
    int offset;
    unsigned int bit;
    for(int i=0; i<npoints; i++){
        for(int j=0; j<codelen; j++){
            table_id = j / 32;
            offset = i*codelen + j;
            bit = (*(unsigned int*)(result_matrix + offset)) >> 31; // get sign bit of a float number
            bit = bit << (j%32);
            codes[table_id*npoints + i] |= bit;
        }
    }
}

__global__
void kernel_binarize_v1(float* result_matrix, unsigned int* codes, int npoints, int ntables, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int start = index*32;
    if(start >= N)
        return;
    int len = min(32, N-start);
    int table_id = index % ntables;
    int point_id = index / ntables;

    unsigned int bit = 0;
    unsigned int val;
    for(int i=0;i<len;i++){
        val = (*(unsigned int*)(result_matrix + start + i)) >> 31;
        bit = bit | (val << i);
    }
    codes[table_id*npoints + point_id] |= bit;
}

// void binarize_gpu(float* d_result_matrix, unsigned int* codes, int npoints, int ntables){
//     unsigned int * d_codes;
//     unsigned int  size = npoints * ntables;
//     unsigned int mem_size = sizeof(unsigned int) * size;

//     checkCudaErrors(cudaMalloc((void **) &d_codes, mem_size));
//     checkCudaErrors(cudaMemset(d_codes, 0, mem_size));

//     // Run kernel on the GPU
//     int N = npoints * ntables * 32; // size of result matrix
//     int blockSize = 1024;
//     int numBlocks = (N/32 + blockSize - 1) / blockSize;
//     printf("numBlocks %d, blockSize %d, %d, %d\n", numBlocks, blockSize, npoints*ntables, N);

//     kernel_binarize_v1<<<numBlocks, blockSize>>>(result_matrix, d_codes, npoints, ntables, N);

//     checkCudaErrors(cudaDeviceSynchronize());
//     // copy result from device to host
//     printf("mem_size %u\n", mem_size);
//     checkCudaErrors(cudaMemcpy(codes, d_codes, mem_size, cudaMemcpyDeviceToHost));
// }


void generate_random_matrix_projection_cpu(float * matrix_projection, int size) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < size; i++) {
        matrix_projection[i] = distribution(generator);
    }
}

void generate_random_matrix_projection(float * matrix_projection, int size) {
    /* Create pseudo-random number generator */
    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    /* Generate n floats on host */
    curandGenerateNormal(gen, matrix_projection, size, 0.0, 1.0);

    /* Cleanup */
    curandDestroyGenerator(gen);
}


void init_matrix_size(sMatrixSize &matrix_size, size_t npoints, int dim, int codelen){
    matrix_size.uiHA = npoints;
    matrix_size.uiWA = dim;

    matrix_size.uiHB = dim;
    matrix_size.uiWB = codelen;

    matrix_size.uiHC = npoints;
    matrix_size.uiWC = codelen;

}

void matrix_multiply_cuda(float* h_A, float* h_B, float* h_C, sMatrixSize &matrix_size){
    float *d_A, *d_B, *d_C;
    // compute memory size
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    // copy host memory to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // setup execution parameters
    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // execute the kernel, CUBLAS version 2.0
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    //note cublas is column primary! need to transpose the order!
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));

    // clean up device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

}


void build_index_gpu(float* h_A, float* h_B, unsigned* codes, sMatrixSize &matrix_size){
    float *d_A, *d_B, *d_C;
    unsigned int * d_codes;
    // compute memory size
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    unsigned int npoints = matrix_size.uiHC;
    unsigned int ntables = matrix_size.uiWC/32;
    unsigned int size_codes =  npoints * ntables;
    unsigned int mem_size_codes = sizeof(unsigned int) * size_codes;

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    checkCudaErrors(cudaMalloc((void **) &d_codes, mem_size_codes));

    // init or copy host memory to device memory
    checkCudaErrors(cudaMemset(d_codes, 0, mem_size_codes));
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // setup execution parameters
    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // execute the kernel, CUBLAS version 2.0
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    //matrix multiplication note cublas is column primary! need to transpose the order!
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

    // Run binarize kernel on the GPU
    int blockSize = 1024;
    int numBlocks = (size_codes + blockSize - 1) / blockSize;
    kernel_binarize_v1<<<numBlocks, blockSize>>>(d_C, d_codes, npoints, ntables, size_C);
    checkCudaErrors(cudaDeviceSynchronize());

    // copy result
    checkCudaErrors(cudaMemcpy(codes, d_codes, mem_size_codes, cudaMemcpyDeviceToHost));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));

    // clean up device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

}

bool verify_correctness(float *matrix_data, float *matrix_projection, float *matrix_result, sMatrixSize &matrix_size){
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *reference = (float *) malloc(mem_size_C);

    matrixMulCPU(reference, matrix_data, matrix_projection, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    // check result (CUBLAS)
    bool resCUBLAS = sdkCompareL2fe(reference, matrix_result, size_C, 1.0e-6f);

    if (resCUBLAS != true)
        printDiff(reference, matrix_result, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    else
        printf("verify correctness succeed!\n");
    delete [] reference;
    return resCUBLAS;
}

void build_index(float* matrix_data, size_t npoints, int dim, int ntables, unsigned int *&codes, float*&matrix_projection){
    int codelen = ntables * 32;
    sMatrixSize matrix_size;

    init_matrix_size(matrix_size, npoints, dim, codelen);
    generate_random_matrix_projection(matrix_projection, codelen * dim);
    build_index_gpu(matrix_data, matrix_projection, codes, matrix_size);

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


int main(int argc, char** argv) {
    // parse argument
    if (argc != 4) {cout << argv[0] << " data_file index_file tableNum)" << endl; exit(-1);}
    char* data_file = argv[1];
    char* index_file = argv[2];
    int ntables = atoi(argv[3]);
    if (ntables < 0 || ntables * 32 > 100000) {cout << "tableNum error!"; exit(-1);}

    // load data
    float* matrix_data = NULL;
    size_t npoints;
    int dim;
    load_data(data_file, matrix_data, npoints, dim);

    // build index
    unsigned int codelen = ntables * 32;
    float *matrix_projection = new float[codelen * dim];
    unsigned int *codes = new unsigned int [npoints * ntables];
    memset(codes, 0, sizeof(unsigned int)*npoints*ntables);

    auto s = std::chrono::high_resolution_clock::now();
    build_index(matrix_data, npoints, dim, ntables, codes, matrix_projection);
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    std::cout << "indexing time: " << diff.count() << "\n";

    saveIndex(index_file, codes, matrix_projection, dim, ntables, npoints);
    return 0;
}
