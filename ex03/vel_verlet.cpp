#include<cmath>
#include<cuda_runtime.h>
__device__ double ljpot(
    const double &eps, const double &sig, 
    const double &dis
){
    double temp;
    temp = 2*pow(sig/dis,6)-1;
    temp *= 24*eps*pow(sig,6)/pow(dis,7);
    return temp;
}
__global__ void calForce(
    double *x, double *f, 
    const int N, const int dim,
    const double epsilon, const double sigma
){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    double f_tot = 0;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i!=j){
                double dis = distance(&x[i*dim],&x[j*dim],dim);
                f_tot += ljpot(epsilon, sigma, dis);
                for(int k=0; k<dim; k++)
                    f[i*dim] += f_tot*(x[i*dim]-x[j*dim])/dis;
            }
        }
    }
}