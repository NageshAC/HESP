#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>
#include "param.cpp"
#include "input.cpp"
#include "vel_verlet.cpp"
#include "output.cpp"

using namespace std;

void cudasafe(int error, string message, string file, int line) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}

int main(){
        
    //reading input parameters
    string paramFileName ="blocks.par",input_path = "../Question/input/";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    
    // reading .par file
    {
        // in param.cpp file
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

    // declearing host vector memory
    int N, dim, frames = (timeEnd/timeStep)+1;  
    // frames -> # of timeframes
    // N -> # of particles
    // dim -> dimension of vector  
    thrust::host_vector<double> sliced;

    readInput(input_path + part_input_file,sliced,N,dim); // in input.cpp
    host_vector<double> x(frames*N*dim,0), v(frames*N*dim,0),f(frames*N*dim,0), m(N,0);

    // extracting m,x,v data from sliced
    // initiating fource
    {
        extract(
            raw_pointer_cast(&x[0]),
            raw_pointer_cast(&v[0]),
            raw_pointer_cast(&m[0]),
            raw_pointer_cast(sliced.data()), N, dim
        );
        // outInput(
        //     raw_pointer_cast(x.data()),
        //     raw_pointer_cast(v.data()),
        //     raw_pointer_cast(m.data()),
        //     N,dim
        // ); // in input.cpp
        
        for(int i=0; i<N; i++)
            calforce(
                raw_pointer_cast(&f[i*dim]), 
                raw_pointer_cast(&x[i*dim]),
                raw_pointer_cast(&x[0]),
                N, dim, epsilon, sigma
            );
        
        // for (int j=0; j<6; j++) cout<<f[j]<<"  ";
        // cout<<endl;
    }
    
    // CUDA PROGRAMMING

    device_vector<double> d_x(x), d_v(v), d_m(m), d_f(f);
    // for (int j=0; j<6; j++) cout<<d_f[j]<<"  ";
    // cout<<endl;
    cudaDeviceProp deviceProp;
    cudasafe(cudaGetDeviceProperties(&deviceProp, 0), "Get Device Properties", __FILE__, __LINE__);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    // cout<<"Max thread per Block = " << maxThreadsPerBlock<<endl;

    int block = (maxThreadsPerBlock);
    int grid = (N/maxThreadsPerBlock)+1;
    // cout<< block<< " : "<<grid<<endl;
    
    for(int i=0; i<frames-1; i++)
        vel_verlet<<<grid,block>>>(
            raw_pointer_cast(&d_x[i*N*dim]),
            raw_pointer_cast(&d_v[i*N*dim]), 
            raw_pointer_cast(&d_f[i*N*dim]),
            raw_pointer_cast(&d_m[0]), N, dim, 
            timeStep, epsilon, sigma
        );
        
    cudasafe(
        cudaDeviceSynchronize(),
        "sync threads", 
        __FILE__, __LINE__
    );
    
    x = d_x; v = d_v; f = d_f;

    // for (int j=1*N*dim; j<1*N*dim+6; j++) cout<<f[j]<<"  ";
    //     cout<<endl;

    // for (int j=1*N*dim; j<1*N*dim+6; j++) cout<<x[j]<<"  ";
    //     cout<<endl;

    // for (int j=1*N*dim; j<1*N*dim+6; j++) cout<<v[j]<<"  ";
    //     cout<<endl;
    
    writeOut(
        part_out_name_base, 
        part_out_freq, 
        raw_pointer_cast(&m[0]),
        raw_pointer_cast(&x[0]),
        raw_pointer_cast(&v[0]),
        frames, N, dim      
    );

    writeVTK(
        vtk_out_name_base,
        vtk_out_freq,
        raw_pointer_cast(&m[0]),
        raw_pointer_cast(&x[0]),
        raw_pointer_cast(&v[0]),
        frames, N, dim
    );

    cout<<"\n\nAll done!\n\n";
        
    return 0;
}
