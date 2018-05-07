#include <iostream>
#include <fstream>

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void load_index(char* filename,  unsigned int *&codes, float *&matrix_projection, unsigned int &dim, unsigned int &ntable, unsigned int &npoint);
void saveIndex(char* filename, unsigned int *codes, float *matrix_projection, unsigned int dim, unsigned int ntables, unsigned int npoints);
void load_data(char* filename, float*& data, unsigned int& num, unsigned int& dim); 
void saveKNNResults(char* filename, unsigned int *result, int nquery, int K);
void init_matrix_size(sMatrixSize &matrix_size, size_t npoints, int dim, int codelen);
