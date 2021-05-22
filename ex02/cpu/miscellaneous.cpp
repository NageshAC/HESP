#pragma once
#include<vector>

using namespace std;

void setTime(vector<double>& t,double& timeEnd, double& timeStep ){
    double val=0;
    while(val <= timeEnd){
        t.push_back(val);
        val+=timeStep;
    }
};
void setProp(vector<vector<vector<double>>>& f, const int size, const int N, const int dim ){
    vector<double> temp_1d;
    vector<vector<double>> temp_2d;
    for(int i=0; i<dim; i++) temp_1d.push_back(0.);
    for(int i=0; i<N; i++) temp_2d.push_back(temp_1d);
    for(int i=0; i<size; i++) f.push_back(temp_2d);
}
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
