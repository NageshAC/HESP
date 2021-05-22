#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
using namespace std;

void writeOut(
    string fileName, int outFreq, vector<double>& m,
    vector<vector<vector<double>>>& x,vector<vector<vector<double>>>& v
    ){
        int frames = x.size(), N = m.size(), dim = x[0][0].size();
        string ffName;
        fstream file;

        for(int i=0; i*outFreq<=frames; i++){

            ffName = "./out/" + fileName + to_string(i) + ".out";
            file.open(ffName, ios::out);
            if(file.is_open()){
                file << N ;

                for(int j=0; j<N; j++){
                    file << "\n";
                    file << setprecision(6) << fixed << m[j] << " ";
                    for(int k=0; k<dim; k++)
                        file << setprecision(6) << fixed << x[i][j][k] << " ";

                    for(int k=0; k<dim; k++)
                        file << setprecision(6) << fixed << v[i][j][k] << " ";
                    
                    
                }
            
                file.close();
            }
            else{
                cout<<"The .out file cannot be opened.\n";
                exit(202);
            }
        }


}
