#pragma once
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>

using namespace std;

void readInput(
        const string& infile, vector<double>& x, vector<double>& v, 
        vector<double>& m, unsigned& N, unsigned& dim){

    fstream ifile;
    ifile.open(infile,ios::in);
    if(ifile.is_open()){
        // cout<<"Input file is opened.\n";

        string out;
        double dTemp;
        vector<string> sliced;
        ifile >> out;
        stringstream(out) >> N;

        while(!ifile.eof()){
            ifile >> out;
            // cout<<out<<" ";
            sliced.push_back(out);
        }
        sliced.pop_back(); // for some reason it is adding 1 extra <last element> at the end
        unsigned line_size = sliced.size()/N;
        dim = (line_size-1)/2;

        // for(int i=0; i<sliced.size(); i++) cout<<sliced[i]<<"  ";

        // cout<<"\nslice size = "<<sliced.size()<<endl;
        // cout<<"Line size = "<<line_size<<endl;
        // cout<<"dim = "<<dim<<endl;


        for(int i=0; i<N; i++){
            for (int j =0; j<dim; j++){
                int x_slice = i*line_size+j;
                int v_slice = i*line_size+dim+j;
                int idx = i*dim+j;
                
                stringstream(sliced[x_slice]) >> dTemp;
                x.push_back(dTemp);
                stringstream(sliced[v_slice]) >> dTemp;
                v.push_back(dTemp);
            }
            stringstream(sliced[(i+1)*line_size-1]) >> dTemp;
            m.push_back(dTemp);
        }
        // cout<<"\nx = ";
        // for(int i=0; i<x.size(); i++) cout<<x[i]<<"  ";
        // cout<<"\nv = ";
        // for(int i=0; i<v.size(); i++) cout<<v[i]<<"  ";
        // cout<<"\nm = ";
        // for(int i=0; i<m.size(); i++) cout<<m[i]<<"  ";

        ifile.close();
        // cout<<"\nInput file closed.\n";
    }
    else{
        cout<<"The input file cannot be opened.\n";
        exit(202);
    }
    
}

void outInput(
        vector<double>& x, vector<double>& v, 
        vector<double>& m, unsigned& N, unsigned& dim){

    cout<<"\n\n--- Printing input file ---\n"<<N<<endl;
    cout<<"\nx = ";
    for(int i=0; i<x.size(); i++) cout<<x[i]<<"  ";
    cout<<"\nv = ";
    for(int i=0; i<v.size(); i++) cout<<v[i]<<"  ";
    cout<<"\nm = ";
    for(int i=0; i<m.size(); i++) cout<<m[i]<<"  ";
    cout<<"\n\n---------------------------\n"<<N<<endl;
}
