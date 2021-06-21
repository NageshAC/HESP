#pragma once
#include<iostream>
#include<cmath>
#include <cuda_runtime.h>
#include"miscellaneous.cpp"

using namespace thrust;

__device__ __host__ double force(
    const double& dis, const double& eps,
    const double& sig
){
    //calculates total force, not components
    double f;
    f = 2*pow(sig/dis, 6)-1;
    f *= 24*eps*pow(sig,6)/pow(dis,7);
    return f;
}
__device__ __host__ void ljpot(
    double* f, double* x, double* y,
    const int& dim, const double& epsilon, const double& sigma
){
    double dis = distance(x,y,dim);
    double fc = force(dis, epsilon, sigma);
    for(int i=0;i<dim;i++){
        f[i] += fc*(x[i]-y[i])/dis;
    }
}

__host__ __device__ void calforce(
    double* f_new,double* x_new, double* x_new_begin, 
    const int& N, const int& dim, 
    double& epsilon, double& sigma
){
    
    for(int i=0; i<N; i++){
        if(x_new == x_new_begin){
            x_new_begin += dim;
            continue;
        }
        else{
            ljpot(f_new, x_new, x_new_begin, dim, epsilon, sigma);
            x_new_begin +=dim;
        }

    }


}
__device__ __host__ void calX( 
    double* x_new, double* x, double* v, double* f,
    const double* m, const double& timeStep,
    const int& dim
){
    double temp = (pow(timeStep,2)/(2*(*m)));
    for(int i=0;i<dim;i++){
        x_new[i] = x[i] + timeStep*v[i] + temp*f[i]; 
    }
}
__device__ void calV(
    double* v_new, double* v, double* f_new,
    double* f, const double* m, 
    const double& timeStep, const int& dim
){
    double temp = timeStep/(2*(*m));
    for(int i=0;i<dim;i++){
        v_new[i] = v[i] + (f[i] + f_new[i])*temp;
    }
}

__global__ void vel_verlet(
    double* x, double* v, double* f,
    const double* m, const int N, 
    const int dim, const double timeStep, 
    double epsilon, double sigma
){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx<N){
        int frame_size = N*dim;
        int part_size = idx*dim;
        double* x_new_begin = x+frame_size;
        double* x_new = x_new_begin + part_size;
        x += part_size;
        v += part_size;
        double* v_new = v+frame_size;
        f += part_size;
        double* f_new = f+frame_size;
        m += idx;
        calX(
            x_new, x, v, f, m, 
            timeStep, dim
        );
        __syncthreads;
        calforce(
            f_new,x_new,x_new_begin,
            N,dim,epsilon,sigma
        );
        calV(
            v_new, v, f_new, f, 
            m, timeStep, dim
        );


    }
}

