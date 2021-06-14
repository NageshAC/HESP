#include<cmath>
#include<cuda_runtime.h>
#include "cell_particle.cpp"

__device__ double ljpot(
    const double &eps, const double &sig, 
    const double &dis
){
    double temp;
    temp = 2*pow(sig/dis,6)-1;
    temp *= 24*eps*pow(sig,6)/pow(dis,7);
    return temp;
}
__global__ void calF(
    cell* cl, particleptr pcl, 
    unsigned& x_n, unsigned& y_n, unsigned& z_n,
    const int N, const int dim,
    const double epsilon, const double sigma
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){

        double f_tot = 0;
        // int line, plane;
        unsigned cell_id = (pcl[idx].head)->get_cell_id();
        for(int i=-1; i<2; i++){
            // plane = cell_id%(x_n*y_n)+i;
            for(int j=-1; j<2; j++){
                // line = cell_id%(x_n)+j;
                for(int k=-1; k<2; k++){
                    // if((i*x_n*y_n*j*x_n*k)%(x_n*y_n)=plane)
                    int cl_id = cell_id+(i*x_n*y_n*j*x_n*k);
                    if(cl_id < x_n*y_n*z_n && cl_id > 0 ){
                        particleptr temp = cl[cl_id].get_even_ptr();
                        while(temp!=NULL){

                        }
                    }

                }
            }
        }



        // for(int j=0; j<N; j++){
        //     if(idx!=j){
        //         double dis = pcl[idx].distance(pcl[j]);
        //         f_tot += ljpot(epsilon, sigma, dis);
        //         for(int k=0; k<dim; k++)
        //             f[idx*dim+k] = f_tot*(x[idx*dim+k]-x[j*dim+k])/dis;
        //     }
        // }
    }
    
}

__global__ void calX(
    double* x, const double* v, 
    const double* f, const double* m,
    const double timestep, const int N, const int dim
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        for(int i=0; i<dim; i++)
            x[idx*dim+i]+=timestep*v[idx*dim+i]+f[idx*dim+i]*pow(timestep,2)/(2*m[idx]);
    }
}

__global__ void calV(
    double* v, const double* f, 
    const double* f_old, const double* m,
    const double timestep, const int N, const int dim
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        for(int i=0; i<dim; i++)
            v[idx*dim+i]+=(f_old[idx*dim+i]+f[idx*dim+i])*timestep/(2*m[idx]);
    }
}