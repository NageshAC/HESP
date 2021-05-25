#pragma once
#include<iostream>
#include<fstream>
#include<sstream>
#include<thrust/host_vector.h>
#include<thrust/fill.h>
#include<cuda_runtime.h>

using namespace thrust;

void setTime(vector<double>& t,double& timeEnd, double& timeStep ){
    double val=0;
    while(val <= timeEnd){
        t.push_back(val);
        val+=timeStep;
    }
}
template <typename T>
__host__ void init3d(T& x){
    for(int i=0;i<x.size(); i++)
        for(int j=0;j<x[0].size(); j++)
            for(int k=0;k<x[0][0].size(); k++)
                x[i][j][k] = 0.;
}
template <typename T>
__host__ void init2d(host_vector<T>& x){
    host_vector<T> t1d(x[0].size(), 0.);
    fill(x.begin(), x.end(),t1d);
}
template <typename T>
__host__ void init1d(host_vector<T>& x){
    fill(x.begin(), x.end(), 0.);
}


///////////////////////////////////////////////////
inline vector<double> add (vector<double>& x,const vector<double>& y){
    // add 2 vector<double>
    vector<double> temp;
    for(int i=0; i<x.size();i++) temp.push_back(x[i] + y[i]);
    return temp;
}
inline vector<double> add (vector<double>& x, vector<double>& y, const vector<double>& z){
    return (add(x, add(y,z)));
}
inline vector<double> mul (vector<double>& x,const double m){
    vector<double> temp;
    for(int i=0; i<x.size();i++) temp.push_back(x[i] * m);
    return temp;
}
inline vector<double> muldiv (vector<double>& x,const double m, const double n){
    if(n==0) cout<<"Division by zero is encountered.\n";
    vector<double> temp = mul(x,m);
    temp = mul(temp,1/n);
    return temp;
}
inline double dis(const vector<double>& x,const vector<double>& y){
    double dis = 0;
    for(int i=0; i<x.size(); i++) dis += pow(x[i]-y[i],2);
    dis = sqrt(dis);
    // cout<<"dis = "<<dis<<endl;
    return dis;
}
