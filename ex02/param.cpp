#pragma once
#include<iostream>
#include<fstream>
#include<sstream>

using namespace std;

void readParam(
    const string& paramFileName,
    string& part_input_file,string &part_out_name_base,string &vtk_out_name_base,
    double& timeStep,double& timeEnd,double& epsilon,double& sigma,
    unsigned& part_out_freq,unsigned& vtk_out_freq,unsigned& cl_wg_1dsize

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
        }
        cout<<"Done reading paramerters file.\n\n";
    }
    else{
        cout<<"The input file cannot be opened.\n";
        exit(202);
    }
    paramFile.close();
}

void outParam(
    const string& part_input_file,const string& part_out_name_base,const string& vtk_out_name_base,
    double& timeStep,double& timeEnd,double& epsilon,double& sigma,
    unsigned& part_out_freq,unsigned& vtk_out_freq,unsigned& cl_wg_1dsize

){
    cout<<"\n--- printing parameters ---\n"
        <<"\npart_input_file "<< part_input_file
        <<"\ntimestep_length "<<timeStep
        <<"\ntime_end "<<timeEnd
        <<"\nepsilon "<< epsilon
        <<"\nsigma "<<sigma
        <<"\npart_out_freq "<<part_out_freq
        <<"\npart_out_name_base "<<part_out_name_base
        <<"\nvtk_out_freq "<<vtk_out_freq
        <<"\nvtk_out_name_base "<<vtk_out_name_base
        <<"\ncl_workgroup_1dsize "<<cl_wg_1dsize
        <<"\n------------------------------\n";

}
