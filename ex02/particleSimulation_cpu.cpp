#include<iostream>
#include<string>
#include<vector>
#include"param.cpp"
#include"input.cpp"
using namespace std;



int main(){

    //reading input parameters
    string paramFileName = "attract.par";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    readParam(
        paramFileName,
        part_input_file, part_out_name_base, vtk_out_name_base,
        timeStep, timeEnd, epsilon, sigma,
        part_out_freq, vtk_out_freq, cl_wg_1dsize
    );
    // outParam(
    //     part_input_file, part_out_name_base, vtk_out_name_base,
    //     timeStep, timeEnd, epsilon, sigma,
    //     part_out_freq, vtk_out_freq, cl_wg_1dsize
    // );

    // initialising the setup;
    unsigned N , dim;
    vector<double> x, v, m, f, fold;
    readInput(part_input_file,x,v,m,N,dim); 
    // init(f , N * dim); init(fold , N * dim);
    outInput(x,v,m,N,dim);


    // delete x,v,m;
    // delete[] f,fold;
    return 0;
}


