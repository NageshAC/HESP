#pragma once
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>

using namespace std;

void outInput(
        vector<vector<double>>& x, vector<vector<double>>& v, 
        vector<double>& m, unsigned& N, unsigned& dim){

    cout<<"\nx = ";
    for(int i=0; i < N; i++)
        for(int j=0;j<dim; j++)cout<<x[i][j]<<"  ";
    cout<<"\nv = ";
    for(int i=0; i < N; i++)
        for(int j=0;j<dim; j++)cout<<v[i][j]<<"  ";
    cout<<"\nm = ";
    for(int i=0; i<m.size(); i++) cout<<m[i]<<"  ";
    cout<<"\n\n";
}

void readInput(
        const string& infile, vector<vector<vector<double>>>& x, 
        vector<vector<vector<double>>>& v, 
        vector<double>& m, unsigned& N, unsigned& dim){

    fstream ifile;
    ifile.open(infile,ios::in);
    if(ifile.is_open()){
        // cout<<"Input file is opened.\n";

        string out;
        double dTemp;
        vector<double> x_temp, v_temp;
        vector<vector<double>> x_2d, v_2d;
        vector<string> sliced;
        ifile >> out;
        stringstream(out) >> N;
        // cout<<"N = "<<N<<"\n";

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

        // x.reserve(N);v.reserve(N);
        for(int i=0; i<N; i++){
            x_temp.clear(); v_temp.clear();
            for (int j =0; j<dim; j++){
                int x_slice = i*line_size+j+1;
                int v_slice = i*line_size+dim+j+1;
                // cout<<x_slice<<" : "<<v_slice<<"\n";
                stringstream(sliced[x_slice]) >> dTemp;
                x_temp.push_back(dTemp);
                stringstream(sliced[v_slice]) >> dTemp;
                v_temp.push_back(dTemp);
            }
            int m_slice = i*line_size;
            // cout<<m_slice<<"\n";
            x_2d.push_back(x_temp); v_2d.push_back(v_temp);
            stringstream(sliced[m_slice]) >> dTemp;
            m.push_back(dTemp);
            // cout<<"m = "<<m[i]<<endl;
        }
        x.push_back(x_2d); v.push_back(v_2d);

        ifile.close();
        // cout<<"\nInput file closed.\n";
    }
    else{
        cout<<"The .in file cannot be opened.\n";
        exit(202);
    }
    
}
