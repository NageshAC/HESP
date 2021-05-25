#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<sstream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>
#include "param.cpp"
#include "input.cpp"
#include "vel_verlet.cpp"
using namespace std;

void cudasafe(int error, string message, string file, int line) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}

int main(){
        
    //reading input parameters
    string paramFileName ="stable.par",input_path = "../Question/input/";
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
    extract(
        x,v,m,
        raw_pointer_cast(sliced.data()), N, dim
    );
    // outInput(
    //     raw_pointer_cast(x.data()),
    //     raw_pointer_cast(v.data()),
    //     raw_pointer_cast(m.data()),
    //     N,dim
    // ); // in input.cpp

    // CUDA PROGRAMMING

    device_vector<double> d_x(x), d_v(v), d_m(m) d_f(frames*N*dim,0);



    cudaDeviceProp deviceProp;
    cudasafe(cudaGetDeviceProperties(&deviceProp, 0), "Get Device Properties", __FILE__, __LINE__);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    cout<<"Max thread per Block = " << maxThreadsPerBlock<<endl;

    return 0;
}
