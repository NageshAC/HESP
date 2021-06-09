#pragma once
#include<iostream>
#include<string>
#include<fstream>
#include<cuda_runtime.h>
#include<iomanip>
using namespace std;

__host__ void writeOut(
    string fileName, int file_count,
    const double* m, double* x, double* v,
    const int& N, const int& dim
){
    fstream file;
    string ffName;
    
    ffName = "./out/" + fileName + to_string(file_count) + ".out";
    file.open(ffName, ios::out);

    if (file.is_open()){
        file << N;
        for(int j=0; j<N; j++){
            file<<setprecision(6)<<std::fixed<<"\n"<<m[j];
            for(int k=0; k<dim; k++){
                file<<setprecision(6)<<std::fixed<<" "<<x[j*dim+k];
            }
            for(int k=0; k<dim; k++){
                file<<setprecision(6)<<std::fixed<<" "<<v[j*dim+k];
            }
        }
        file.close();

    }
    else{
        cout<<"The .out file cannot be opened.\n";
        exit(202);
    }
}