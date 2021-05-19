#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<cstdio>
using namespace std;

template<typename T>
void init(T* f, unsigned N){
    T* a = new T [N];
    for(int i=0; i<N; i++) a[i] = 0;
    f = a;
}

void readInput(
        string infile, double *x, double *v, 
        double *m, unsigned* N, unsigned* dim){
    // opening input file
    fstream ifile;
    ifile.open(infile,ios::in);
        
    if (ifile.is_open()){ // checking if the input file is open
        cout<<"Input file opened.\n";
        // declearing few parameters
        string out;
        vector<string> sliced; // holds seperated string

        ifile >> out;
        stringstream(out) >> *N; // first line is always number of particles
        cout<<"N = "<<*N<<endl;

        // for(int i=0; i<*N && !ifile.eof();i++){
        //     cout<<"entered i loop.\n";
        //     for(int j=0; !ifile.eof(); j++){
        //         ifile >> out;
        //         if (out == "\n" && i==0){
        //             *dim = (j-1)/2;
        //             cout<<"dim calculated: dim = "<<*dim<<endl;
        //             init(x,*N * *dim);init(v,*N * *dim);init(m,*N); // initiating x, v, m
        //             cout<<"x,v,m initated.\n";
        //             break;
        //         }
        //         if (out == "\n" || ifile.eof()) break;
        //         sliced.push_back(out);

        //     }
            
        //     // extracting x, v and m info
        //     for(int j=0; j<*dim; j++){
        //         stringstream(sliced[j]) >> x[i*(*dim)+j];
        //         stringstream(sliced[j+3]) >> v[i*(*dim)+j];
        //     }
        //     stringstream(sliced[-1]) >> m[i];
        // }
        ifile.close();
        cout<<"Input file closed.\n";
    }
    else{
        string str;
        cout<<"The input file cannot be opened.\nPress any key to exit.......\n";
        getline(cin,str);
        exit(202);
    }
    
}

void writeInput(
        string infile, double *x, double *v, 
        double *m, unsigned N, unsigned dim){

    cout<<"\n\nPrinting input file.\n\n"<<N<<endl;
    for(int j=0; j<N; j++){
        for(int i=0; i<dim; i++) cout<<x[j*(dim)+i]<<" ";
        for(int i=0; i<dim; i++) cout<<v[j*(dim)+i]<<" ";
        cout<<m[j]<<endl;
    }
}

void readParam(
    string part_input_file,string part_out_name_base,string vtk_out_name_base,
    double* timeStep,double* timeEnd,double* epsilon,double* sigma,
    unsigned* part_out_freq,unsigned* vtk_out_freq,unsigned* cl_wg_1dsize

){

}

int main(){

    //reading input parameters
    // string part_input_file, part_out_name_base, vtk_out_name_base;
    // double timeStep, timeEnd, epsilon, sigma;
    // unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    string part_input_file= "attract.in";
    // readParam(
    //     part_input_file, part_out_name_base, vtk_out_name_base,
    //     &timeStep, &timeEnd, &epsilon, &sigma,
    //     &part_out_freq, &vtk_out_freq, &cl_wg_1dsize
    // );

    // initialising the setup;
    unsigned N , dim;
    double *x, *v, *m;
    // double *f, *fold;
    readInput(part_input_file,x,v,m,&N,&dim); 
    // init(f,N * dim); init(fold,N * dim);
    // writeInput(part_input_file,x,v,m,N,dim);


    delete[] x,v,m;
    // delete[] f,fold;
    return 0;
}


