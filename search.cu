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

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

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


#define PROFILE 1

static std::chrono::_V2::system_clock::time_point ticker[6];
static double timer[6] = {0};
static const int ENCODE_QUERY_MATRIX = 0;
static const int COMPUTE_HAMMING_DISTANCE = 1;
static const int HAMMING_DISTANCE_SORTING = 2;
static const int COMPUTE_EUCLIDEAN_DISTANCE = 3;
static const int EUCLIDEAN_DISTANCE_SORTING = 4;
static const int GET_RESULT = 5;
#if PROFILE
#define START_ACTIVITY(X) ticker[X]=std::chrono::high_resolution_clock::now();
#define END_ACTIVITY(X) timer[X]+=(static_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now()-ticker[X])).count()
#define PRINT_PROFILER cout<<"encode query matrix:"<<timer[0]<<endl<<"compute hamming distance:"<<timer[1]<<endl<<"hamming distance sorting:"<<timer[2]<<endl<<"compute euclidean distance:"<<timer[3]<<endl<<"euclidean distance sorting:"<<timer[4]<<endl<<"get result:"<<timer[5]<<endl;
#else
#define START_ACTIVITY(X)
#define END_ACTIVITY(X)
#define PRINT_PROFILER
#endif


#define SQUARE4(v) v.x *= v.x; v.y *= v.y; v.z *= v.z; v.w *= v.w;
#define MINUS4(v1, v2) v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; v1.w -= v2.w;
#define REDUCE4(v) v.x += v.y; v.x += v.z; v.x += v.w;



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
}

__global__
void kernel_hamming_distance(unsigned int* d_query_codes, unsigned int* d_base_codes, unsigned int* d_hamming_distance, int nbase, int nquery, int ntable, unsigned int* d_hamming_distance_idx){
    unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(base_idx>=nbase || query_idx>=nquery || query_idx*nbase + base_idx >= (nquery*nbase))
        return;

    int base_start = 0;
    int query_start = 0;
    unsigned int x;
    unsigned int result=0;
    for (size_t i = 0; i < ntable; i++) {
        x = d_base_codes[base_start+base_idx] ^ d_query_codes[query_start+query_idx];
        //compute number of 1 in x
        x = (x & 0x55555555) + ((x >> 1 ) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2 ) & 0x33333333);
        x = (x & 0x0f0f0f0f) + ((x >> 4 ) & 0x0f0f0f0f);
        x = (x & 0x00ff00ff) + ((x >> 8 ) & 0x00ff00ff);
        x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff);

        result += x;
        base_start += nbase;
        query_start += nquery;
    }
    //d_hamming_distance[query_idx*nbase + base_idx] = result;
    d_hamming_distance[query_idx*nbase + base_idx] = result + (query_idx<<16);
    //d_hamming_distance[query_idx*nbase + base_idx] = result;
    d_hamming_distance_idx[query_idx*nbase + base_idx] = base_idx;
}



//TODO optimize IO
__global__
void kernel_euclidean_distance(float* d_query_matrix, float* d_base_matrix, unsigned int* d_hamming_distance_idx, double* d_euclidean_distance,  int nquery, int nbase, int L, int dim, float min_value, float max_value, unsigned int* d_euclidean_distance_idx){
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int L_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(query_idx>=nquery || L_idx>=L)
        return;

    int base_idx = d_hamming_distance_idx[query_idx*nbase + L_idx];
    double result = 0;
    int query_start = query_idx*dim;
    int base_start = base_idx*dim;
    double diff;
    for(int i=0;i<dim;i++){
        diff = d_query_matrix[query_start+i] - d_base_matrix[base_start+i];
        result += diff * diff;
    }
    //normalize result, so result is < 1
    result = result / (max_value*max_value*dim);
    d_euclidean_distance[query_idx*L + L_idx] = result + query_idx;
    d_euclidean_distance_idx[query_idx*L + L_idx] = base_idx;
}

__global__
void kernel_euclidean_distance_v2(float* d_query_matrix, float* d_base_matrix, unsigned int* d_hamming_distance_idx, double* d_euclidean_distance,  int nquery, int nbase, int L, int dim, float min_value, float max_value, unsigned int* d_euclidean_distance_idx){
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int L_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(query_idx>=nquery || L_idx>=L)
        return;

    int base_idx = d_hamming_distance_idx[query_idx*nbase + L_idx];
    int query_start = query_idx*dim/4;
    int base_start = base_idx*dim/4;

    float4 query_point;
    float4 base_point;
    double result = 0;
    for(int i=0;i<dim/4;i++){
        query_point = reinterpret_cast<float4*>(d_query_matrix)[query_start+i];
        base_point = reinterpret_cast<float4*>(d_base_matrix)[base_start+i];

        MINUS4(query_point, base_point);
        SQUARE4(query_point);
        REDUCE4(query_point);
        result += query_point.x;
    }
    //normalize result, so result is < 1
    result = result / (max_value*max_value*dim);
    d_euclidean_distance[query_idx*L + L_idx] = result + query_idx;
    d_euclidean_distance_idx[query_idx*L + L_idx] = base_idx;
}


__global__
void kernel_get_result(unsigned int* d_euclidean_distance_idx, unsigned int* d_result, int nquery, int K, int L){
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(query_idx>=nquery || K_idx>=K)
        return;

    d_result[query_idx*K + K_idx] = d_euclidean_distance_idx[query_idx*L+ K_idx];
}


void compute_index_gpu(float* d_A, float* d_B, unsigned int* d_codes, sMatrixSize &matrix_size, int npoint, int ntable){
    // alloc device memory to store projection result
    float *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

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

    dim3 threadsPerBlock(1024/ntable, ntable);
    dim3 numBlocks((npoint + threadsPerBlock.x -1) / threadsPerBlock.x, (ntable+threadsPerBlock.y-1) / threadsPerBlock.y);
    kernel_binarize<<<numBlocks, threadsPerBlock>>>(d_C, d_codes, npoint, ntable);
    checkCudaErrors(cudaDeviceSynchronize());

    // clean up device memory
    checkCudaErrors(cudaFree(d_C));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));
}


void knn_prepare(float *&d_projection_matrix, unsigned int* &d_base_codes, float *&d_base_matrix, float *projection_matrix, unsigned int *base_codes, float *base_matrix, int dim, int ntable, int nbase){
    int codelen = ntable * 32;
    
    unsigned int size_projection_matrix = dim * codelen;
    unsigned int mem_size_projection_matrix = sizeof(float) * size_projection_matrix;
    unsigned int size_base_codes = nbase * ntable;
    unsigned int mem_size_base_codes = sizeof(unsigned int) * size_base_codes;
    unsigned int size_base_matrix = nbase * dim;
    unsigned int mem_size_base_matrix = sizeof(float) * size_base_matrix;

    checkCudaErrors(cudaMalloc((void **) &d_projection_matrix, mem_size_projection_matrix));
    checkCudaErrors(cudaMalloc((void **) &d_base_codes, mem_size_base_codes));
    checkCudaErrors(cudaMalloc((void **) &d_base_matrix, mem_size_base_matrix));

    checkCudaErrors(cudaMemcpy(d_projection_matrix, projection_matrix, mem_size_projection_matrix, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_base_codes, base_codes, mem_size_base_codes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_base_matrix, base_matrix, mem_size_base_matrix, cudaMemcpyHostToDevice));

}

void knn_free(float *&d_projection_matrix, unsigned int* &d_base_codes, float *&d_base_matrix){
    checkCudaErrors(cudaFree(d_projection_matrix));
    checkCudaErrors(cudaFree(d_base_codes));
    checkCudaErrors(cudaFree(d_base_matrix));
}


void knn_search(unsigned int* result, float *d_base_matrix, float *query_matrix, float *d_projection_matrix, unsigned int* d_base_codes, unsigned int dim, unsigned int ntable, unsigned int nbase, unsigned int nquery, int L, int K, float min_value, float max_value){

    //device memory
    float *d_query_matrix;
    double *d_euclidean_distance;
    unsigned int *d_query_codes, *d_hamming_distance, *d_hamming_distance_idx, *d_euclidean_distance_idx, *d_result;
    int codelen = ntable * 32;

    // ============== Step1: Encodes query matrix =====================
    // this step will compute d_C = d_A * d_B and binarize d_C to d_codes

    // prepare d_projection_matrix and d_query code
    //Input: d_query_matrix
    START_ACTIVITY(ENCODE_QUERY_MATRIX);

    unsigned int size_query_matrix = nquery * dim;
    unsigned int mem_size_query_matrix = sizeof(float) * size_query_matrix;
    checkCudaErrors(cudaMalloc((void **) &d_query_matrix, mem_size_query_matrix));
    checkCudaErrors(cudaMemcpy(d_query_matrix, query_matrix, mem_size_query_matrix, cudaMemcpyHostToDevice));

    //Output: d_query_codes
    unsigned int size_query_codes = nquery * ntable;
    unsigned int mem_size_query_codes = sizeof(unsigned int) * size_query_codes;
    checkCudaErrors(cudaMalloc((void **) &d_query_codes, mem_size_query_codes));
    checkCudaErrors(cudaMemset(d_query_codes, 0, mem_size_query_codes));

    sMatrixSize matrix_size;
    init_matrix_size(matrix_size, nquery, dim, codelen);
    compute_index_gpu(d_query_matrix, d_projection_matrix, d_query_codes, matrix_size, nquery, ntable);

    END_ACTIVITY(ENCODE_QUERY_MATRIX);

    //============== Step2: Compute Hamming Distance =====================
    //Input: d_query_codes, d_base_codes
    //Output: d_hamming_distance, d_hamming_distance_idx
    //Memory Cost: memcost = 2 * base_size * 1M (if nquery=1000, memcost=2G)

    //Input: d_base_codes
    START_ACTIVITY(COMPUTE_HAMMING_DISTANCE);

    //Output: hamming_distance
    unsigned int size_hamming_distance = nquery * nbase;
    unsigned int mem_hamming_distance  = sizeof(unsigned int) * size_hamming_distance;
    checkCudaErrors(cudaMalloc((void **) &d_hamming_distance, mem_hamming_distance));

    //Onput: d_hamming_distance_idx
    unsigned int size_hamming_distance_idx = nquery * nbase;
    unsigned int mem_hamming_distance_idx  = sizeof(unsigned int) * size_hamming_distance_idx;
    checkCudaErrors(cudaMalloc((void **) &d_hamming_distance_idx, mem_hamming_distance_idx));

    dim3 threadsPerBlock(1, 1024);
    dim3 numBlocks((nquery + threadsPerBlock.x -1) / threadsPerBlock.x, (nbase+threadsPerBlock.y-1) / threadsPerBlock.y);
    //kernel_hamming_distance_v1<<<numBlocks, threadsPerBlock>>>(d_query_codes, d_base_codes, d_hamming_distance, nbase, nquery, ntable);
    kernel_hamming_distance<<<numBlocks, threadsPerBlock>>>(d_query_codes, d_base_codes, d_hamming_distance, nbase, nquery, ntable, d_hamming_distance_idx);
    checkCudaErrors(cudaDeviceSynchronize());

    //Useless: d_query_codes, d_base_codes,
    checkCudaErrors(cudaFree(d_query_codes));

    END_ACTIVITY(COMPUTE_HAMMING_DISTANCE);


    //============== Step3: Sort According to Hamming Distance  =====================
    START_ACTIVITY(HAMMING_DISTANCE_SORTING);
    thrust::device_ptr<unsigned int> d_ptr_keys = thrust::device_pointer_cast(d_hamming_distance);
    thrust::device_ptr<unsigned int> d_ptr_values = thrust::device_pointer_cast(d_hamming_distance_idx);

    thrust::sort_by_key(d_ptr_keys, d_ptr_keys + size_hamming_distance, d_ptr_values);

    //Useless: d_hamming_distance
    checkCudaErrors(cudaFree(d_hamming_distance));
    END_ACTIVITY(HAMMING_DISTANCE_SORTING);

    //============== Step4: Compute Euclidean Distance Bwtween Real Feature Vector =====================
    //Input: d_query_matrix, d_base_matrix, d_hamming_distance_idx
    //Output: d_euclidean_distance
    //Memory Cost: memcost = d_hamming_distance_idx + d_base_matrix = 1G + 1G, if nquery=1000, dim=960
    //Input: d_base_matrix

    START_ACTIVITY(COMPUTE_EUCLIDEAN_DISTANCE);

    //Output: d_euclidean_distance
    unsigned int size_euclidean_distance = nquery * L;
    unsigned int mem_size_euclidean_distance = sizeof(double) * size_euclidean_distance;
    checkCudaErrors(cudaMalloc((void **) &d_euclidean_distance, mem_size_euclidean_distance));

    //Onput: d_euclidean_distance_idx
    unsigned int size_euclidean_distance_idx = nquery * L;
    unsigned int mem_size_euclidean_distance_idx  = sizeof(unsigned int) * size_euclidean_distance_idx;
    checkCudaErrors(cudaMalloc((void **) &d_euclidean_distance_idx, mem_size_euclidean_distance_idx));

    threadsPerBlock.x = 1;
    threadsPerBlock.y = 1024;
    numBlocks.x = (nquery + threadsPerBlock.x -1) / threadsPerBlock.x;
    numBlocks.y = (L+threadsPerBlock.y-1) / threadsPerBlock.y;
    kernel_euclidean_distance_v2<<<numBlocks, threadsPerBlock>>>(d_query_matrix, d_base_matrix, d_hamming_distance_idx, d_euclidean_distance, nquery, nbase, L, dim, min_value, max_value, d_euclidean_distance_idx);
    checkCudaErrors(cudaDeviceSynchronize());
    //Useless
    checkCudaErrors(cudaFree(d_query_matrix));
    checkCudaErrors(cudaFree(d_hamming_distance_idx));

    END_ACTIVITY(COMPUTE_EUCLIDEAN_DISTANCE);


    //============== Step5: Sort Base According to Euclidean Distance =====================
    START_ACTIVITY(EUCLIDEAN_DISTANCE_SORTING);
    thrust::device_ptr<double> d_ptr_keys_2 = thrust::device_pointer_cast(d_euclidean_distance);
    thrust::device_ptr<unsigned int> d_ptr_values_2 = thrust::device_pointer_cast(d_euclidean_distance_idx);
    thrust::sort_by_key(d_ptr_keys_2, d_ptr_keys_2 + size_euclidean_distance, d_ptr_values_2);

    //Useless: d_euclidean_distance
    checkCudaErrors(cudaFree(d_euclidean_distance));

    END_ACTIVITY(EUCLIDEAN_DISTANCE_SORTING);

    //============== Step6: Get Result  =====================
    //Onput: d_result
    START_ACTIVITY(GET_RESULT);
    unsigned int size_result = nquery * K;
    unsigned int mem_size_result  = sizeof(unsigned int) * size_result;
    checkCudaErrors(cudaMalloc((void **) &d_result, mem_size_result));

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    numBlocks.x = (nquery + threadsPerBlock.x -1) / threadsPerBlock.x;
    numBlocks.y = (K+threadsPerBlock.y-1) / threadsPerBlock.y;
    kernel_get_result<<<numBlocks, threadsPerBlock>>>(d_euclidean_distance_idx, d_result, nquery, K, L);
    checkCudaErrors(cudaDeviceSynchronize());

    //move result to CPU
    checkCudaErrors(cudaMemcpy(result, d_result, mem_size_result, cudaMemcpyDeviceToHost));

    //free memory
    checkCudaErrors(cudaFree(d_result));
    checkCudaErrors(cudaFree(d_euclidean_distance_idx));

    END_ACTIVITY(GET_RESULT);
}


int main(int argc, char** argv){

    float *base_matrix = NULL;
    float *query_matrix = NULL;
    float *projection_matrix = NULL;
    unsigned int *base_codes = NULL;
    unsigned int *result = NULL;

    unsigned int nbase=0;
    unsigned int nquery=0;
    unsigned int ntable=0;
    unsigned int base_dim=0;
    unsigned int query_dim=0;

    float *d_base_matrix, *d_projection_matrix;
    unsigned int * d_base_codes;

    // parse argument
    if(argc!=9){cout<< argv[0] << " index_file data_file query_file result_file ntable initsz querNN batch_size )" <<endl; exit(-1);}
    char* index_file = argv[1];
    char* base_file = argv[2];
    char* query_file = argv[3];
    char* result_file = argv[4];
    ntable = atoi(argv[5]);
    int L = atoi(argv[6]); // retrieve L points from index
    int K = atoi(argv[7]); // return K points at the end using real feature
    int batch_size = atoi(argv[8]);

    // load data and query
    load_data(base_file, base_matrix, nbase, base_dim);
    load_data(query_file, query_matrix, nquery, query_dim);

    // load index
    result = new unsigned int [nquery*K];
    load_index(index_file, base_codes, projection_matrix, base_dim, ntable, nbase);


    float min_value_query = *std::min_element(query_matrix, query_matrix + nquery*query_dim);
    float max_value_query = *std::max_element(query_matrix, query_matrix + nquery*query_dim);
    float min_value_base = *std::min_element(base_matrix, base_matrix + nbase * base_dim);
    float max_value_base = *std::max_element(base_matrix, base_matrix + nbase * base_dim);

    cout << "data load okay," << endl;
    std::cout << "min query " << min_value_query << '\n';
    std::cout << "max query " << max_value_query << '\n';
    std::cout << "min data " << min_value_base << '\n';
    std::cout << "max data " << max_value_base << '\n';

    // knn search
    cout << "begin knn search, batch size " << batch_size << endl;
    int n_remain = nquery;
    int n_completed = 0;
    auto s = std::chrono::high_resolution_clock::now();

    
    knn_prepare(d_projection_matrix, d_base_codes, d_base_matrix, projection_matrix, base_codes, base_matrix, base_dim, ntable, nbase);

    while(n_remain>0){
        int tmp_nquery = min(batch_size, n_remain);
        knn_search(result+n_completed*K, d_base_matrix, query_matrix + n_completed*query_dim, d_projection_matrix, d_base_codes, base_dim, ntable, nbase, tmp_nquery, L, K, min(min_value_base, min_value_query), max(max_value_base, max_value_query));
        n_remain -= tmp_nquery;
        n_completed += tmp_nquery;

    }
    auto e = std::chrono::high_resolution_clock::now();

    // report time
    std::chrono::duration<double> diff = e-s;
    std::cout << "query searching time: " << diff.count() << "\n";

    PRINT_PROFILER;
    saveKNNResults(result_file, result, nquery, K);


    knn_free(d_projection_matrix, d_base_codes, d_base_matrix);
    delete [] projection_matrix;
    delete [] base_codes;
    delete [] base_matrix;

    return 0;
}
