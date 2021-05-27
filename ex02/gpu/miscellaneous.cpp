#pragma once
#include<iostream>
#include<cmath>

using namespace std;

__device__ __host__ double distance (
    double *a, double* b,
    const int& dim
){
    double dis = 0;
    for(int i=0;i<dim;i++){
        dis += pow(a[i]-b[i],2); 
    }
    return sqrt(dis);
}
