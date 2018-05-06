#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

void load_data(char* filename, unsigned int*& data, unsigned int& num, unsigned int& dim) { // load data with sift10K pattern
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
    data = new unsigned int[num * dim];
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
    cout << "load data okay, npoints " << num << ", dim " << dim << endl;
}


void evaluate(unsigned int *gt_matrix, unsigned int *predicted_matrix, unsigned int K, unsigned int nquery){
    set<unsigned int> s1;
    set<unsigned int> s2;
    vector<unsigned int> common_data;
    int num_common = 0;
    for(size_t i=0;i<nquery;i++){
        s1.clear(); s2.clear(); common_data.clear();
        s1 = set<unsigned int>(gt_matrix + i*K, gt_matrix + (i+1)*K);
        s2 = set<unsigned int>(predicted_matrix + i*K, predicted_matrix + (i+1)*K);
        set_intersection(s1.begin(),s1.end(),s2.begin(),s2.end(), std::back_inserter(common_data));
        num_common += common_data.size();
        if(i%200 == 0)
            cout << "i " << i << " Recall " << num_common*1.0/((i+1)*K) << endl;
    }
    printf("Recall %f\n", num_common*1.0/(K*nquery));
}


int main(int argc, char** argv){
    unsigned int gt_nquery;
    unsigned int gt_dim;
    unsigned int predicted_nquery;
    unsigned int predicted_dim;
    unsigned int *predicted_matrix, *gt_matrix;
    // parse argument
    if(argc!=3){cout<< argv[0] << "ground truth file, result file" <<endl; exit(-1);}
    char *gt_file = argv[1];
    char *predicted_file = argv[2];

    // load data and query
    load_data(gt_file, gt_matrix, gt_nquery, gt_dim);
    load_data(predicted_file, predicted_matrix, predicted_nquery, predicted_dim);

    //evaluatin
    evaluate(gt_matrix, predicted_matrix, predicted_dim, predicted_nquery);
    return 0;
}
