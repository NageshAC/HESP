
#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>
#include "../param.cpp"
#include "../input.cpp"
#include "vel_verlet.cpp"
#include "../output.cpp"

using namespace std;

void cudasafe(int error, string message, string file, int line) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}

int main(){
        
    //reading input parameters
    string paramFileName ="blocks_big.par",input_path = "../Question/input/";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;

    unsigned x_n, y_n, z_n, cl_wg_3dsize_x, cl_wg_3dsize_y, cl_wg_3dsize_z;
    double x_min, x_max, y_min, y_max, z_min, z_max,
         r_cut, r_skin;
    
    // reading .par file
    {
        // in param.cpp file
        readParam(
            input_path + paramFileName,
            part_input_file, part_out_name_base, vtk_out_name_base,
            timeStep, timeEnd, epsilon, sigma,
            part_out_freq, vtk_out_freq, cl_wg_1dsize,
            cl_wg_3dsize_x, cl_wg_3dsize_y, cl_wg_3dsize_z,
            x_min, x_max, y_min, y_max, z_min, z_max,
            x_n, y_n, z_n, r_cut, r_skin
        );
        // outParam(
        //     part_input_file, part_out_name_base, 
        //     vtk_out_name_base, timeStep, timeEnd, epsilon, sigma,
        //     part_out_freq, vtk_out_freq, cl_wg_1dsize,
        //     cl_wg_3dsize_x, cl_wg_3dsize_y, cl_wg_3dsize_z,
        //     x_min, x_max, y_min, y_max, z_min, z_max,
        //     x_n, y_n, z_n, r_cut, r_skin
        // );
    }
    
    double del_x=(x_max-x_min)/x_n, del_y=(y_max-y_min)/y_n, del_z=(z_max-z_min)/z_n;
    // declearing host vector memory
    unsigned N, dim, frames = (timeEnd/timeStep);  
    // frames -> # of timeframes
    // N -> # of particles
    // dim -> dimension of vector  
    thrust::host_vector<double> sliced;

    readInput(input_path + part_input_file,sliced,N,dim); // in input.cpp
    host_vector<double> x(N*dim,0), v(N*dim,0), m(N,0),f(N*dim,0) ;

    // extracting m,x,v data from sliced
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
    }
    // CUDA Programming
    device_vector<double> d_x(x),d_v(v),d_f(N*dim,0),d_f_old(N*dim,0),zeros(N*dim,0),d_m(m), neighbourhood(N*N,0);
    cudaDeviceProp deviceProp;
    cudasafe(
        cudaGetDeviceProperties(&deviceProp,0),
        "Get device Properties",
        __FILE__, __LINE__
    );
    // cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<endl;

    int blockSize = deviceProp.maxThreadsPerBlock, 
    gridSize = (N/deviceProp.maxThreadsPerBlock)+1;

    // cout<<"Block Size: "<<blockSize<<"\nGrid size: "<<gridSize<<endl
    // <<"Frames: "<<frames<<endl;

    // create neighbourhood list

    calN<<<gridSize, blockSize>>>(
        raw_pointer_cast(&neighbourhood[0]),
        raw_pointer_cast(&d_x[0]),
        r_cut, N, dim
    );

    // Initial force calculation
    calF<<<gridSize, blockSize>>>(
        raw_pointer_cast(&d_x[0]),
        raw_pointer_cast(&d_f[0]),
        raw_pointer_cast(&neighbourhood[0]),
        N, dim, epsilon, sigma, r_cut
    );
    cudasafe(
        cudaDeviceSynchronize(),
        "sync threads", 
        __FILE__, __LINE__
    );
    writeOut(
        part_out_name_base, 0,
        raw_pointer_cast(&m[0]),
        raw_pointer_cast(&x[0]),
        raw_pointer_cast(&v[0]),
        N, dim
    ); // in output.cpp
    writeVTK(
        vtk_out_name_base, 0,
        raw_pointer_cast(&m[0]),
        raw_pointer_cast(&x[0]),
        raw_pointer_cast(&v[0]),
        N, dim
    ); // in output.cpp

    for(int i=1; i<=frames; i++){

        calX<<<gridSize, blockSize>>>(
            raw_pointer_cast(&d_x[0]),
            raw_pointer_cast(&d_v[0]),
            raw_pointer_cast(&d_f[0]),
            raw_pointer_cast(&d_m[0]), 
            timeStep, N, dim,
            x_min, x_max, y_min, 
            y_max, z_min, z_max
        );
        cudasafe(
            cudaDeviceSynchronize(),
            "sync threads", 
            __FILE__, __LINE__
        );

        d_f_old = d_f;
        thrust::copy(zeros.begin(), zeros.end(), d_f.begin());
        
        calN<<<gridSize, blockSize>>>(
            raw_pointer_cast(&neighbourhood[0]),
            raw_pointer_cast(&d_x[0]),
            r_cut, N, dim
        );

        calF<<<gridSize, blockSize>>>(
            raw_pointer_cast(&d_x[0]),
            raw_pointer_cast(&d_f[0]),
            raw_pointer_cast(&neighbourhood[0]),
            N, dim, epsilon, sigma, r_cut
        );

        cudasafe(
            cudaDeviceSynchronize(),
            "sync threads", 
            __FILE__, __LINE__
        );

        calV<<<gridSize,blockSize>>>(
            raw_pointer_cast(&d_v[0]),
            raw_pointer_cast(&d_f[0]),
            raw_pointer_cast(&d_f_old[0]),
            raw_pointer_cast(&d_m[0]),
            timeStep, N, dim
        );

        cudasafe(
            cudaDeviceSynchronize(),
            "sync threads", 
            __FILE__, __LINE__
        );
        
        if(i%part_out_freq == 0){
            m = d_m; x = d_x; v = d_v;
            writeOut(
                part_out_name_base, (i/part_out_freq),
                raw_pointer_cast(&m[0]),
                raw_pointer_cast(&x[0]),
                raw_pointer_cast(&v[0]),
                N, dim
            ); // in output.cpp
        }
        if(i%vtk_out_freq == 0){
            m = d_m; x = d_x; v = d_v;
            writeVTK(
                part_out_name_base, (i/vtk_out_freq),
                raw_pointer_cast(&m[0]),
                raw_pointer_cast(&x[0]),
                raw_pointer_cast(&v[0]),
                N, dim
            ); // in output.cpp
        }
    }

    std::cout<<"\n\nAll done!\n\n";
        
    return 0;
}