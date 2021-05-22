#include<iostream>
#include<string>
#include<vector>
#include<cmath>
#include"param.cpp"
#include"input.cpp"
#include"output.cpp"
// #include"miscellaneous.cpp"
#include"vel_ver.cpp"

using namespace std;


int main(){
    //reading input parameters
    string paramFileName ="stable.par",input_path = "./Question/input/";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    
    // reading .par file
    {   // in param.cpp file
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
    unsigned N , dim, size;
    vector<vector<vector<double>>> x, v, f;
    vector<double> t, m;

    setTime(t, timeEnd, timeStep); // in miscellaneous.cpp file
    size = t.size();
    t.shrink_to_fit();
    
    // reading .in file and initiating x,v,f
    {
        // reading .in file
        readInput(input_path + part_input_file,x,v,m,N,dim);    // in input.cpp

        m.resize(N); m.shrink_to_fit();

        // initiating with 0, x,v already have 1 2d layer from input so size-1
        setProp(x,size-1,N,dim);
        setProp(v,size-1,N,dim);    // in miscellaneous.cpp file
        setProp(f,size,N,dim);

        // outInput(x[0],v[0],m,N,dim); // in input.cpp
    }
    
    // initial force calculation
    for(int i=0; i<N; i++)f[0][i]=(calForce(x[0],i, epsilon, sigma));
    // in vel_ver.cpp

    // vel verlet
    for(int i=1;i<size;i++){
        for(int j=0;j<N; j++){
            vel_ver_x(x,v,f,m,timeStep,i,j);  // in vel_ver.cpp
        }
        for(int j=0;j<N; j++){
            vel_ver_vf(x,v,f,m,epsilon,sigma,timeStep,i,j);  // in vel_ver.cpp
        }
    }

    writeOut(part_out_name_base, part_out_freq, m, x, v);  // in output.cpp

    cout<<"All done!";

    return 0;
}


