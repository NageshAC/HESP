#include<cmath>
#include<cuda_runtime.h>
// #include "../miscellaneous.cpp"
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
    double *x, double *f, double *n,
    const int N, const int dim,
    const double epsilon, 
    const double sigma, 
    const double r_cut
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx<N){

        double f_tot = 0;
        for(int j=0; j<N; j++){
            if(n[idx*N+j]>0){
                f_tot = ljpot(epsilon, sigma, n[idx*N+j]);
                for(int k=0; k<dim; k++)
                    f[idx*dim+k] += f_tot*(x[idx*dim+k]-x[j*dim+k])/n[idx*N+j];
            }
        }
    }    
}

__global__ void calX(
    double* x, const double* v, 
    const double* f, const double* m,
    const double timestep, const int N, const int dim,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const double z_min, const double z_max
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        double pos_min[3] = { x_min, y_min, z_min };
        double pos_max[3] = { x_max, y_max, z_max };
        for(int i=0; i<dim; i++){
            x[idx*dim+i]+=timestep*v[idx*dim+i]+f[idx*dim+i]*pow(timestep,2)/(2*m[idx]);

            if(x[idx*dim+i]>pos_max[i])
                x[idx*dim+i] += pos_min[i]-pos_max[i];
            if(x[idx*dim+i]<pos_min[i])
            x[idx*dim+i] += pos_max[i]-pos_min[i];
        }
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


__global__ void calN(
    double* n, double* x, 
    const double r_cut,
    const int N, const int dim
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        for(int j=idx+1; j<N; j++){
            double dis = distance(&x[idx*dim],&x[j*dim],dim);
            if(dis<=r_cut){
                n[idx*N+j] = dis;
                n[j*N+idx] = dis;
            }
            
        }
    }
}