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
    string paramFileName ="attract.par",input_path = "./Question/input/";
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
    host_vector<double> x(N*dim,0), v(N*dim,0),f(N*dim,0), m(N,0);

    // extracting m,x,v data from sliced
    // initiating fource
    {
        extract(
            raw_pointer_cast(&x[0]),
            raw_pointer_cast(&v[0]),
            raw_pointer_cast(&m[0]),
            raw_pointer_cast(sliced.data()), N, dim
        );
        outInput(
            raw_pointer_cast(x.data()),
            raw_pointer_cast(v.data()),
            raw_pointer_cast(m.data()),
            N,dim
        ); // in input.cpp
        
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
    


    cout<<"\n\nAll done!\n\n";
        
    return 0;
}