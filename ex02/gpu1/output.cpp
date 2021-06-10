#pragma once
#include<iostream>
#include<string>
#include<fstream>
#include<cuda_runtime.h>
#include<iomanip>
using namespace std;

__host__ void writeOut(
    string fileName, const int& out_freq,
    const double* m, const double* x, const double* v,
    const int& frames, const int& N, const int& dim
){
    fstream file;
    int file_count = 0;
    string ffName;

    for(int i=0; i<=frames; i+=out_freq){
        ffName = "./out/" + fileName + to_string(file_count) + ".out";
        file.open(ffName, ios::out);

        if (file.is_open()){
            file_count++;
            file << N;
            for(int j=0; j<N; j++){
                file<<setprecision(6)<<std::fixed<<"\n"<<m[j];
                for(int k=0; k<dim; k++){
                    file<<setprecision(6)<<std::fixed<<" "<<x[i*N*dim+j*dim+k];
                }
                for(int k=0; k<dim; k++){
                    file<<setprecision(6)<<std::fixed<<" "<<v[i*N*dim+j*dim+k];
                }
            }
            file.close();

        }
        else{
            cout<<"The .out file cannot be opened.\n";
            exit(202);
        }
    }
}

__host__ void writeVTK(
    string fileName, const int& out_freq,
    const double* m,  const double* x, const double* v,
    const int& frames, const int& N, const int& dim
){
    fstream file;
    int file_count = 0;
    string ffName;

    string vtk_version = "# vtk DataFile Version 4.0",
        comments = "HESPA simulation",
        file_type = "ASCII",
        dataset = "DATASET UNSTRUCTURED_GRID";


    for(int i=0; i<=frames; i+=out_freq){
        ffName = "./out/" + fileName + to_string(file_count) + ".vtk";
        file.open(ffName, ios::out);

        if (file.is_open()){
            file_count++;

            // vtk file header
            file<<vtk_version<<endl<<comments<<endl<<file_type<<endl
                <<dataset<<endl<<"POINTS "<<N<<" double"<<endl;
            // x points
            for(int j=0; j<N; j++){
                for(int k=0; k<dim; k++)
                    file<<setprecision(6)<<std::fixed<<x[i*N*dim+j*dim+k]<<" ";
                file<<endl;
            }

            // m points
            file<<"CELLS 0 0\nCELL_TYPES 0\nPOINT_DATA "<<N
                <<"\nSCALARS m double\nLOOKUP_TABLE default\n";
            for(int j=0; j<N; j++)
                file<<setprecision(6)<<std::fixed<<m[j]<<endl;
            
            // v data
            file<<"VECTORS v double\n";
            for(int j=0; j<N; j++){
                for(int k=0; k<dim; k++)
                    file<<setprecision(6)<<std::fixed<<v[i*N*dim+j*dim+k]<<" ";
                file<<endl;
            }

            file.close();

        }
        else{
            cout<<"The .vtk file cannot be opened.\n";
            exit(202);
        }
    }
}