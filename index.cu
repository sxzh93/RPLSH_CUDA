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

#include "util.h"

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


// Compute reference data set matrix multiply on CPU
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

__global__
void kernel_binarize(float* result_matrix, unsigned int* codes, int npoints, int ntables)
{
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    int table_id = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_id >= npoints || table_id >= ntables)
        return;

    unsigned int codelen = 32*ntables;
    unsigned int bit = 0;
    unsigned int result = 0;
    unsigned int column_start = table_id*32;
    for(int i=0;i<32;i++){
        if(result_matrix[point_id*codelen + column_start + i] > 0){
            bit = 1u << i;
            result |= bit;
        }
    }
    codes[table_id*npoints+point_id] = result;
    //codes[point_id*ntables + table_id] = result;
}


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
    unsigned int codelen = ntables * 32;
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
    float *h_C = new float[npoints*codelen];
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    float sum = 0;
    for(unsigned int i=0;i<npoints*codelen;i++)
        sum += h_C[i];
    printf("projection result sum %f\n", sum);

    // Run binarize kernel on the GPU
    //int blockSize = 1024;
    //int numBlocks = (size_codes + blockSize - 1) / blockSize;

    dim3 threadsPerBlock(1024/ntables, ntables);
    dim3 numBlocks((npoints + threadsPerBlock.x -1) / threadsPerBlock.x, (ntables+threadsPerBlock.y-1) / threadsPerBlock.y);

    kernel_binarize<<<numBlocks, threadsPerBlock>>>(d_C, d_codes, npoints, ntables);
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


void build_index(float* matrix_data, size_t npoints, int dim, int ntables, unsigned int *&codes, float*&matrix_projection){
    int codelen = ntables * 32;
    sMatrixSize matrix_size;

    init_matrix_size(matrix_size, npoints, dim, codelen);
    //generate_random_matrix_projection(matrix_projection, codelen * dim);
    generate_random_matrix_projection_cpu(matrix_projection, codelen * dim);
    build_index_gpu(matrix_data, matrix_projection, codes, matrix_size);

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
    unsigned int npoints;
    unsigned int dim;
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

    unsigned int sum = 0;
    for(int i=0;i<npoints*ntables;i++)
        sum += codes[i];
    printf("sum %u\n", sum);
    saveIndex(index_file, codes, matrix_projection, dim, ntables, npoints);
    return 0;
}
