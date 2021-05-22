#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include"miscellaneous.cpp"

using namespace std;


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

