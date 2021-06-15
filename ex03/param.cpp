#pragma once
#include<iostream>
#include<fstream>
#include<sstream>

using namespace std;

void readParam(
    const string& paramFileName,
    string& part_input_file,string &part_out_name_base,string &vtk_out_name_base,
    double& timeStep,double& timeEnd,double& epsilon,double& sigma,
    unsigned& part_out_freq,unsigned& vtk_out_freq,unsigned& cl_wg_1dsize,
    unsigned& cl_wg_3dsize_x, unsigned& cl_wg_3dsize_y, unsigned& cl_wg_3dsize_z,
    double& x_min, double& x_max, double& y_min, double& y_max, 
    double& z_min, double& z_max,
    unsigned& x_n, unsigned& y_n, unsigned& z_n, double& r_cut, double& r_skin
){
    fstream paramFile;
    paramFile.open(paramFileName, ios::in);
    if(paramFile.is_open()){
        // cout<<"Parameter file is opened.\n";
        string out1, out2;
        
        while(!paramFile.eof()){
            paramFile >> out1;
            paramFile >> out2;
            // cout<<out1<<"-"<<out2<<endl;
            if(out1 == "part_input_file") part_input_file = out2;
            if(out1 == "timestep_length") stringstream(out2) >> timeStep;
            if(out1 == "time_end") stringstream(out2) >> timeEnd;
            if(out1 == "epsilon") stringstream(out2) >> epsilon;
            if(out1 == "sigma") stringstream(out2) >> sigma;
            if(out1 == "part_out_freq") stringstream(out2) >> part_out_freq;
            if(out1 == "part_out_name_base") part_out_name_base = out2;
            if(out1 == "vtk_out_freq") stringstream(out2) >> vtk_out_freq;
            if(out1 == "vtk_out_name_base") vtk_out_name_base = out2;
            if(out1 == "cl_workgroup_1dsize") stringstream(out2) >> cl_wg_1dsize;

            if(out1 == "cl_workgroup_3dsize_x ") stringstream(out2) >> cl_wg_3dsize_x;
            if(out1 == "cl_workgroup_3dsize_y ") stringstream(out2) >> cl_wg_3dsize_y;
            if(out1 == "cl_workgroup_3dsize_z ") stringstream(out2) >> cl_wg_3dsize_z;
            
            if(out1 == "x_min") stringstream(out2) >> x_min;
            if(out1 == "x_max") stringstream(out2) >> x_max;
            if(out1 == "y_min") stringstream(out2) >> y_min;
            if(out1 == "y_max") stringstream(out2) >> y_max;
            if(out1 == "z_min") stringstream(out2) >> z_min;
            if(out1 == "z_max") stringstream(out2) >> z_max;

            if(out1 == "x_n") stringstream(out2) >> x_n;
            if(out1 == "y_n") stringstream(out2) >> y_n;
            if(out1 == "z_n") stringstream(out2) >> z_n;
            if(out1 == "r_cut") stringstream(out2) >> r_cut;
            if(out1 == "r_skin") stringstream(out2) >> r_skin;
        }
        // cout<<"Done reading paramerters file.\n\n";
        paramFile.close();
    }
    else{
        cout<<"The .par file cannot be opened.\n";
        exit(202);
    }
    
}

void outParam(
    const string& part_input_file,const string& part_out_name_base,const string& vtk_out_name_base,
    const double& timeStep,const double& timeEnd,const double& epsilon,const double& sigma,
    const unsigned& part_out_freq,const unsigned& vtk_out_freq,const unsigned& cl_wg_1dsize,
    const int& cl_wg_3dsize_x, const int& cl_wg_3dsize_y, const int& cl_wg_3dsize_z,
    const double& x_min, const double& x_max, const double& y_min, const double& y_max, 
    const double& z_min, const double& z_max,
    const int& x_n, const int& y_n, int& z_n, const double& r_cut, double& r_skin

){
    cout<<"\n--- printing parameters ---\n"
        <<"\npart_input_file: "<< part_input_file
        <<"\ntimestep_length: "<<timeStep
        <<"\ntime_end: "<<timeEnd
        <<"\nepsilon: "<< epsilon
        <<"\nsigma: "<<sigma
        <<"\npart_out_freq: "<<part_out_freq
        <<"\npart_out_name_base: "<<part_out_name_base
        <<"\nvtk_out_freq: "<<vtk_out_freq
        <<"\nvtk_out_name_base: "<<vtk_out_name_base
        <<"\ncl_workgroup_1dsize: "<<cl_wg_1dsize
        <<"\ncl_workgroup_3dsize_x: "<<cl_wg_3dsize_x
        <<"\ncl_workgroup_3dsize_y: "<<cl_wg_3dsize_y
        <<"\ncl_workgroup_3dsize_z: "<<cl_wg_3dsize_z
        <<"\nx_min: "<<x_min
        <<"\nx_max: "<<x_max
        <<"\ny_min: "<<y_min
        <<"\ny_max: "<<y_max
        <<"\nz_min: "<<z_min
        <<"\nz_max: "<<z_max
        <<"\nx_n: "<<x_n
        <<"\ny_n: "<<y_n
        <<"\nz_n: "<<z_n
        <<"\nr_cut: "<<r_cut
        <<"\nr_skin: "<<r_skin
        <<"\n------------------------------\n";

}
