#include<iostream>
#include<string>
#include<vector>
#include<cmath>
#include"param.cpp"
#include"input.cpp"
#include"output.cpp"

using namespace std;

void setTime(vector<double>& t,double& timeEnd, double& timeStep ){
    double val=0;
    while(val <= timeEnd){
        t.push_back(val);
        val+=timeStep;
    }
};
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

vector<double>calForce(vector<vector<double>>& x, int n,
    const double& eps, const double& sig){
    vector<double> f;
    for(int j=0; j<x[0].size();++j) f.push_back(0.); 
    for(int j=0; j<x.size();j++)
        if(n!=j) f = add(f, ljpot(x[n], x[j], eps, sig));
        // cout<< f[0] <<" ; "<< f[1]<<" ; "<< f[2]<<endl;
    return f;
}

vector<double> calx(
    vector<double>& x, vector<double>& v, vector<double>& f, 
    const double timeStep, const double m){
    vector<double> temp1, temp2;
    temp1 = muldiv(f,pow(timeStep,2),2*m);
    temp2 = mul(v,timeStep);
    temp1 = add(temp1, temp2, x);
    return temp1;
}

vector<double> calv(
    vector<double>& v, vector<double>& fold, 
    vector<double>& f, 
    const double timeStep, const double m
){
    vector<double> temp = add(fold,f);
    temp = muldiv(temp, timeStep,2*m);
    temp = add(temp,v);
    return temp;
}

void vel_ver_x(
    vector<vector<vector<double>>>& x, vector<vector<vector<double>>>& v,
    vector<vector<vector<double>>>& f, vector<double>& m,
    double timeStep, int frame, int n
){
    x[frame][n] = calx(x[frame-1][n], v[frame-1][n], f[frame-1][n], timeStep, m[n]);
};
void vel_ver_vf(
    vector<vector<vector<double>>>& x, vector<vector<vector<double>>>& v,
    vector<vector<vector<double>>>& f, vector<double>& m, double epsilon,
    double sigma, double timeStep, int frame, int n
){
    f[frame][n] = calForce(x[frame],n,epsilon,sigma);
    v[frame][n] = calv(v[frame-1][n],f[frame-1][n],f[frame][n], timeStep, m[n]);
};

void setProp(vector<vector<vector<double>>>& f, const int size, const int N, const int dim ){
    vector<double> temp_1d;
    vector<vector<double>> temp_2d;
    for(int i=0; i<dim; i++) temp_1d.push_back(0.);
    for(int i=0; i<N; i++) temp_2d.push_back(temp_1d);
    for(int i=0; i<size; i++) f.push_back(temp_2d);
}

int main(){
    //reading input parameters
    string paramFileName = "./Question/input/stable.par";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    
    // reading .par file
    readParam(
        paramFileName,
        part_input_file, part_out_name_base, vtk_out_name_base,
        timeStep, timeEnd, epsilon, sigma,
        part_out_freq, vtk_out_freq, cl_wg_1dsize
    );
    // outParam(
    //     part_input_file, part_out_name_base, vtk_out_name_base,
    //     timeStep, timeEnd, epsilon, sigma,
    //     part_out_freq, vtk_out_freq, cl_wg_1dsize
    // );

    // initialising the setup;
    unsigned N , dim;
    vector<vector<vector<double>>> x, v, f;
    vector<double> t, m;
    setTime(t, timeEnd, timeStep);
    int size = t.size();
    t.shrink_to_fit();

    // reading .in file
    readInput(part_input_file,x,v,m,N,dim);

    m.resize(N); m.shrink_to_fit();
    // initiating with 0, x,v already have 1 2d layer from input so size-1
    setProp(x,size-1,N,dim);
    setProp(v,size-1,N,dim);
    setProp(f,size,N,dim);
    outInput(x[0],v[0],m,N,dim);

    for(int i=0; i<N; i++)f[0][i]=(calForce(x[0],i, epsilon, sigma));
    // cout<<endl<<x.size()<<endl;
    // cout<<endl<<x[0].size()<<endl;
    // cout<<endl<<x[0][0].size()<<endl;

    // vel verlet
    for(int i=1;i<size;i++){
        for(int j=0;j<N; j++){
            vel_ver_x(x,v,f,m,timeStep,i,j);
        }
        for(int j=0;j<N; j++){
            vel_ver_vf(x,v,f,m,epsilon,sigma,timeStep,i,j);
        }
    }

    writeOut(part_out_name_base, part_out_freq, m, x, v);

    return 0;
}


