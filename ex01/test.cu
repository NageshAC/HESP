#include<iostream>
#include <cuda_runtime.h>
#include<cmath>
#include <cuda.h>
using namespace std;

int w = 2048, h = w;
int d = 64;

void cudasafe(int error, string message="(---)", string file = "(-this file-)", int line = -1) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}
__global__ void set(int* val,const int w,const int d){
    int idx = blockIdx.x * (w/d)*(w/d) + threadIdx.y * (w/d) + threadIdx.x;
    val[idx] = idx;
}

int main(){
    
    int* d_int;
    int* val = new int [w*h];
    cudasafe(cudaMalloc(&d_int,w*h*sizeof(int)),"Mem Allo",__FILE__,__LINE__);
    dim3 blocks(w/d,h/d);
    dim3 grids(d*d);
    set<<<grids, blocks>>>(d_int, w,d);
    cudasafe(cudaMemcpy(val,d_int,w*h*sizeof(int),cudaMemcpyDeviceToHost),
                "Mem Allo",__FILE__,__LINE__);
    for(int i=0; i<w*h; i++)cout<<val[i]<<"  ";
    cudaFree(d_int);
    delete[] val;
    return 0;
}