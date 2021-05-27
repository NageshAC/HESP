#include<iostream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cstdlib>
#include<algorithm>
#include<cmath>


using namespace thrust;

__global__ void vecAdd(float* a,float* b,float* c,int n){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<n){
        c[idx] = pow(a[idx] + b[idx],2);
    }
}


int main(){
    host_vector<float> a(2048), b(2048);
    const auto generator = []{return rand();};
    std::generate(a.begin(),a.end(), generator);
    std::generate(b.begin(),b.end(), generator);
    device_vector<float> d_a(a), d_b(b), d_c(2048,0) ;
    vecAdd<<<2,1024>>>(
        raw_pointer_cast(&d_a[0]), raw_pointer_cast(&d_b[0]),
        raw_pointer_cast(&d_c[0]), 2048
    );
    // host_vector<float> c(d_c);
    for(int i=0; i<2048; i++) std::cout<<d_c[i]<<"  ";
    std::cout<<std::endl;
    return 0;
}
