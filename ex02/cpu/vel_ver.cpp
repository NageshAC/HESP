#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include"miscellaneous.cpp"

using namespace std;

inline double force(const double& dis, 
const double& eps,const double& sig){
    //calculates total force, not components
    double f;
    f = 2*pow(sig/dis, 6)-1;
    f *= 24*eps*pow(sig,6)/pow(dis,7);
    return f;
}

inline vector<double> ljpot(
    vector<double>& x, vector<double>& y,
    const double& epsilon,const double& sigma){
    vector<double> f;
    double temp = 0, d = dis(x,y);
    for (int i=0; i<x.size();i++){
        temp = force(d,epsilon,sigma); // Total force
        f.push_back(temp*(x[i]-y[i])/d); // force components
    }
    return f;
}

// vector<vector<double>> calForce(
//      vector<vector<double>>& x,
//     const double& eps, const double& sig){
//         vector<vector<double>> f;
//         vector<double> t;
//         t.resize(x[0].size());
//         for(int i=0; i<x.size();i++){
//             for(int j=0; j<x[0].size();++j) t[j]=0;
//             for(int j=0; j<x.size();j++)
//                 if(i!=j) t = add(t, ljpot(x[i], x[j], eps, sig));
//             f.push_back(t);
//         }
//         cout<< f[0][0] <<" ; "<< f[0][1]<<" ; "<< f[0][2]<<endl;
//         cout<< f[1][0] <<" ; "<< f[1][1]<<" ; "<< f[1][2]<<endl;
//     return f;
// }

void calForce(vector<double>& f, vector<vector<double>>& x, int n,
    const double& eps, const double& sig){

    for(int j=0; j<x.size();j++)
        if(n!=j) f = add(f, ljpot(x[n], x[j], eps, sig));
        // cout<< f[0] <<" ; "<< f[1]<<" ; "<< f[2]<<endl;
}

void calx(
    vector<double>& x, vector<double>& x_old, 
    vector<double>& v, vector<double>& f, 
    const double timeStep, const double m
){
    vector<double> temp1, temp2;
    temp1 = muldiv(f,pow(timeStep,2),2*m);
    temp2 = mul(v,timeStep);
    temp1 = add(temp1, temp2, x_old);
    x = temp1;
}

void calv(
    vector<double>& v, vector<double>& v_old, 
    vector<double>& f_old, vector<double>& f, 
    const double timeStep, const double m
){
    v = add(f_old,f);
    v = muldiv(v, timeStep,2*m);
    v = add(v,v_old);
}

void vel_ver_x(
    vector<vector<vector<double>>>& x, vector<vector<vector<double>>>& v,
    vector<vector<vector<double>>>& f, vector<double>& m,
    double timeStep, int frame, int n
){
    calx(x[frame][n], x[frame-1][n], v[frame-1][n], f[frame-1][n], timeStep, m[n]);
};
void vel_ver_vf(
    vector<vector<vector<double>>>& x, vector<vector<vector<double>>>& v,
    vector<vector<vector<double>>>& f, vector<double>& m, double epsilon,
    double sigma, double timeStep, int frame, int n
){
    calForce(f[frame][n], x[frame],n,epsilon,sigma);
    calv(v[frame][n], v[frame-1][n],f[frame-1][n],f[frame][n], timeStep, m[n]);
};

