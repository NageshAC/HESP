#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<cuda_runtime.h>
#include "param.cpp"
#include "input.cpp"
#include "vel_verlet_neighbour.cpp"
#include "output.cpp"
#include "particle.cpp"

using namespace std;

void cudasafe(int error, string message, string file, int line) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}

int main(){
        
    //reading input parameters
    string paramFileName ="blocks_big.par",input_path = "./Question/input/";
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
    unsigned cell_n = x_n*y_n*z_n;
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

    // particle creation
    particleptr pcl = new particle[N];
    for(unsigned i=0; i<N; i++){
        particle temp(
            i, 
            raw_pointer_cast(&x[i*dim]),
            raw_pointer_cast(&v[i*dim]),
            raw_pointer_cast(&f[i*dim]),
            raw_pointer_cast(&m[i])
        );
        temp.calCell_id(
            x_min, del_x,
            y_min, del_y, 
            z_min, del_z,
            x_n, y_n
        );
        pcl[i] = temp;
    }
    // double* pos = pcl[N-1].x;
    // cout<<pos[0]<<"\t"<<pos[1]<<"\t"<<pos[2]<<"\n";

    // cell creation
    // cell* cl = new cell[cell_n];
    // for(unsigned i=0; i<N; i++){
    //     cl[pcl[i].cell_id].addParticle(&pcl[i]);
    // }


    // CUDA Programming
    device_vector<double> d_x(x),d_v(v),d_f(f),d_m(m);
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

    // cout<< "Size of particle: "<<sizeof(particle)<<endl;
    // cout<< "Size of cell: "<<sizeof(cell)<<endl;

    // device particle allocation
    particleptr d_pcl;
    cudasafe(
        cudaMalloc(&d_pcl, N*sizeof(particle)), "Particle Alloc",
        __FILE__, __LINE__
    );
    cudasafe(
        cudaMemcpy(d_pcl, pcl, N*sizeof(particle), cudaMemcpyHostToDevice),
        "Particle Mem cpy", __FILE__, __LINE__
    );

    cell* cl = new cell[cell_n];
    for(unsigned i=0; i<N; i++){
        cl[pcl[i].cell_id].addParticle(&pcl[i]);
    }

    cell* d_cl;
    cudasafe(
        cudaMalloc(&d_cl, cell_n*sizeof(cell)), "cell Alloc",
        __FILE__, __LINE__
    );

    for(int i=0; i<N; i++){
        // d_cl[d_pcl[i].cell_id].last = &d_pcl[i];
        (d_cl[((d_pcl[i]).cell_id)]).last = &(d_pcl[i]);
    }

    // calF<<<gridSize,blockSize>>>(
    //     cl, pcl, 
    //     x_n, y_n, z_n, 
    //     N, dim, 
    //     epsilon, sigma
    // );



    cout<<"\n\nAll done!\n\n";
        
    return 0;
}