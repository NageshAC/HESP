#include<iostream>
#include<vector>
#include<cmath>
#include<string>
#include"param.cpp"
#include"input.cpp"
#include"output.cpp"
// #include"miscellaneous.cpp"
#include<cuda_runtime.h>

using namespace std;

void cudasafe(
    int error, string message="(---)", 
    string file = "(-this file-)", 
    int line = -1
){
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}
void setTime(vector<double>& t,double& timeEnd, double& timeStep ){
    double val=0;
    while(val <= timeEnd){
        t.push_back(val);
        val+=timeStep;
    }
}

void setProp(
    vector<vector<vector<double>>>& f, const int frames, 
    const int N, const int dim 
){
    vector<double> temp_1d;
    vector<vector<double>> temp_2d;
    for(int i=0; i<dim; i++) temp_1d.push_back(0.);
    for(int i=0; i<N; i++) temp_2d.push_back(temp_1d);
    for(int i=0; i<frames; i++) f.push_back(temp_2d);
}

__host__ __device__ __global__ inline vector<double> add (
    vector<double>& x,const vector<double>& y
){
    // add 2 vector<double>
    vector<double> temp;
    for(int i=0; i<x.size();i++) temp.push_back(x[i] + y[i]);
    return temp;
}
__host__ __device__ __global__ inline vector<double> add (
    vector<double>& x, vector<double>& y, const vector<double>& z
){
    return (add(x, add(y,z)));
}
__host__ __device__ __global__ inline vector<double> mul (
    vector<double>& x,const double m
){
    vector<double> temp;
    for(int i=0; i<x.size();i++) temp.push_back(x[i] * m);
    return temp;
}
__host__ __device__ __global__ inline vector<double> muldiv (
    vector<double>& x,const double m, const double n
){
    if(n==0) cout<<"Division by zero is encountered.\n";
    vector<double> temp = mul(x,m);
    temp = mul(temp,1/n);
    return temp;
}
__host__ __device__ __global__ inline double dis(
    const vector<double>& x,const vector<double>& y
){
    double dis = 0;
    for(int i=0; i<x.size(); i++) dis += pow(x[i]-y[i],2);
    dis = sqrt(dis);
    // cout<<"dis = "<<dis<<endl;
    return dis;
}
__host__ __device__ __global__ inline double force(const double& dis, 
    const double& eps,const double& sig
){
        //calculates total force, not components
        double f;
        f = 2*pow(sig/dis, 6)-1;
        f *= 24*eps*pow(sig,6)/pow(dis,7);
        return f;
}  
__host__ __device__ __global__ inline vector<double> ljpot(
    vector<double>& x, vector<double>& y,
    const double& epsilon,const double& sigma
){
    vector<double> f;
    double temp = 0, d = dis(x,y);
    for (int i=0; i<x.size();i++){
        temp = force(d,epsilon,sigma); // Total force
        f.push_back(temp*(x[i]-y[i])/d); // force components
    }
    return f;
}

__host__ __device__ __global__ vector<double>calForce(
    vector<vector<double>>& x, int n,
    const double& eps, const double& sig
){
    vector<double> f;
    for(int j=0; j<x[0].size();++j) f.push_back(0.); 
    for(int j=0; j<x.size();j++)
        if(n!=j) f = add(f, ljpot(x[n], x[j], eps, sig));
        // cout<< f[0] <<" ; "<< f[1]<<" ; "<< f[2]<<endl;
    return f;
}

__host__ __device__ __global__ vector<double> calx(
    vector<double>& x, vector<double>& v, vector<double>& f, 
    const double timeStep, const double m
){
    vector<double> temp1, temp2;
    temp1 = muldiv(f,pow(timeStep,2),2*m);
    temp2 = mul(v,timeStep);
    temp1 = add(temp1, temp2, x);
    return temp1;
}

__host__ __device__ __global__ vector<double> calv(
    vector<double>& v, vector<double>& fold, 
    vector<double>& f, 
    const double timeStep, const double m
){
    vector<double> temp = add(fold,f);
    temp = muldiv(temp, timeStep,2*m);
    temp = add(temp,v);
    return temp;
}

__host__ __device__ __global__ void vel_ver(
    vector<vector<vector<double>>>& x, vector<vector<vector<double>>>& v,
    vector<vector<vector<double>>>& f, vector<double>& m, double epsilon,
    double sigma, double timeStep, int blockSize, int frame
){  
    idx = blockIdx.x * blockSize + threadIdx.x;
    x[frame][idx] = calx(
        x[frame-1][idx], v[frame-1][idx], 
        f[frame-1][idx], timeStep, m[idx]);
    __syncthreads();
    f[frame][idx] = calForce(x[frame],idx,epsilon,sigma);
    v[frame][idx] = calv(v[frame-1][idx],
        f[frame-1][idx],f[frame][idx], timeStep, m[idx]);
}




int main(){
    // reading input parameters
    string paramFileName ="stable.par",input_path = "./Question/input/";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    
    // reading .par file
    {  // in param.cpp file
        readParam(
            input_path + paramFileName,
            part_input_file, part_out_name_base, vtk_out_name_base,
            timeStep, timeEnd, epsilon, sigma,
            part_out_freq, vtk_out_freq, cl_wg_1dsize
        );
        // outParam(
        //     part_input_file, part_out_name_base, 
        //     vtk_out_name_base, timeStep, timeEnd, epsilon, sigma,
        //     part_out_freq, vtk_out_freq, cl_wg_1dsize
        // ); 
    }

    // initialising the setup;
    unsigned N , dim, frames;
    vector<vector<vector<double>>> x, v, f;
    vector<double> t, m;
    

    setTime(t, timeEnd, timeStep); // in miscellaneous.cpp file
    frames = t.size();
    t.shrink_to_fit();
    
    // reading .in file and initiating x,v,f
    {
        // reading .in file
        readInput(input_path + part_input_file,x,v,m,N,dim); // in input.cpp

        m.resize(N); m.shrink_to_fit();

        // initiating with 0, x,v already have 1 2d layer from input so size-1
        setProp(x,frames-1,N,dim);
        setProp(v,frames-1,N,dim);    // in miscellaneous.cpp file
        setProp(f,frames,N,dim);

        // outInput(x[0],v[0],m,N,dim);     // in input.cpp
    }
    
    // programming on CUDA and GPU

    double *d_x, *d_v, *d_f;
    double *d_m;
    double *d_timeStep, *d_sigma, *d_epsilon;
    unsigned *d_frames, *d_N, *d_dim;

    // allocating mem size
    { 
        cudasafe(
            cudaMalloc(&d_x, frames*N*dim*sizeof(double)),
            "x mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_v, frames*N*dim*sizeof(double)),
            "v mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_f, frames*N*dim*sizeof(double)),
            "f mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_m, N*sizeof(double)),
            "m mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_timeStep, sizeof(double)),
            "timestep mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_sigma, sizeof(double)),
            "sigma mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_epsilon, sizeof(double)),
            "epsilon mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_frames, sizeof(unsigned)),
            "frames mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_N, sizeof(unsigned)),
            "N mem alloc", __FILE__, __LINE__
        );
        cudasafe(
            cudaMalloc(&d_dim, sizeof(unsigned)),
            "dim mem alloc", __FILE__, __LINE__
        );

    }

    // copying data from host to device
    {
        cudasafe(
            cudaMemcpy(d_x ,& x[0][0][0], frames*N*dim*sizeof(double), cudaMemcpyHostToDevice), 
            "x mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_v ,& v[0][0][0], frames*N*dim*sizeof(double), cudaMemcpyHostToDevice), 
            "v mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_f ,& f[0][0][0], frames*N*dim*sizeof(double), cudaMemcpyHostToDevice), 
            "f mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_m ,& m[0], N*sizeof(double), cudaMemcpyHostToDevice), 
            "m mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_timeStep,& timeStep, sizeof(double), cudaMemcpyHostToDevice), 
            "timeStep mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_sigma ,& sigma, sizeof(double), cudaMemcpyHostToDevice), 
            "sigma mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_epsilon ,& epsilon, sizeof(double), cudaMemcpyHostToDevice), 
            "epsilon mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_frames ,& frames, sizeof(unsigned), cudaMemcpyHostToDevice), 
            "frames mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_N ,& N, sizeof(unsigned), cudaMemcpyHostToDevice), 
            "N mem copy HostToDevice", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(d_dim ,& dim, sizeof(unsigned), cudaMemcpyHostToDevice), 
            "dim mem copy HostToDevice", __FILE__, __LINE__
        );
    }

    // kernel execution
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        if(N<prop.maxThreadPerBlock){
            dim3 grid (1);
            dim3 block (N);
        }
        else{
            dim3 block(prop.maxThreadPerBlock);
            dim3 grid(ceil(N/prop.maxThreadPerBlock));
        }
        for(int i=1; i<frames; i++)
            vel_ver<<<grid,block>>>(
                d_x, d_v, d_f, d_m, d_epsilon, d_sigma,
                d_timeStep, prop.maxThreadPerBlock, i
            );

    }
    // copying data from device to host
    {
        cudasafe(cudaDeviceSynchronize(),"Sync", __FILE__,__LINE__);
        cudasafe(
            cudaMemcpy(x , d_x, frames*N*dim*sizeof(double), cudaMemcpyDeviceToHost), 
            "x mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(v , d_v, frames*N*dim*sizeof(double), cudaMemcpyDeviceToHost), 
            "v mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(f , d_f, frames*N*dim*sizeof(double), cudaMemcpyDeviceToHost), 
            "f mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(m , d_m, N*sizeof(double), cudaMemcpyDeviceToHost), 
            "m mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(timeStep, d_timeStep, sizeof(double), cudaMemcpyDeviceToHost), 
            "timeStep mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(sigma , d_sigma, sizeof(double), cudaMemcpyDeviceToHost), 
            "sigma mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(epsilon , d_epsilon, sizeof(double), cudaMemcpyDeviceToHost), 
            "epsilon mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(frames , d_frames, sizeof(unsigned), cudaMemcpyDeviceToHost), 
            "frames mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(N , d_N, sizeof(unsigned), cudaMemcpyDeviceToHost), 
            "N mem copy Device to Host", __FILE__, __LINE__
        );
        cudasafe(
            cudaMemcpy(dim , d_dim, sizeof(unsigned), cudaMemcpyDeviceToHost), 
            "dim mem copy Device to Host", __FILE__, __LINE__
        );
    }

    // distroying created variables
    {
        cudasafe(
            cudaFree(d_x), "x mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_v), "v mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_f), "f mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_m), "m mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_timeStep), "timeStep mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_sigma), "sigma mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_epsilon), "epsilon mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_frames), "frames mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_N), "N mem free",
            __FILE__, __LINE__
        );
        cudasafe(
            cudaFree(d_dim), "dim mem free",
            __FILE__, __LINE__
        );
    }
    
    writeOut(part_out_name_base, part_out_freq, m, x, v);  // in output.cpp

    cout<<"All done!";

    return 0;
}

