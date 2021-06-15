#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<thrust/host_vector.h>
#include"miscellaneous.cpp"

using namespace thrust;

void outInput( 
    double* x,
    double* v,
    double* m, const int N, const int dim
){
    cout<<"\n------------------------------------------------------------\n"
        << N;
    for(int i=0; i<N; i++){
        cout<<std::fixed<<std::setprecision(6)<<endl<<m[i];
        for(int j=0; j<dim; ++j)cout<<" "<<x[i*dim+j];
        for(int j=0; j<dim; ++j)cout<<" "<<v[i*dim+j];
    }
    cout<<"\n------------------------------------------------------------\n\n";


}

void readInput( string fileName, 
    host_vector<double>& sliced, unsigned& N, unsigned& dim
){
    fstream file;
    file.open(fileName);
    if (file.is_open()){
        // declearing few extra variables
        double out;

        file >> N;
        // stringstream(out) >> N;

        do{
            file >> out;
            sliced.push_back(out);
        }while(!file.eof());
        sliced.pop_back(); // for some reason it is adding 1 extra <last element> at the end
        
        unsigned line_size = sliced.size()/N;
        dim = (line_size-1)/2;

        // test print
        // cout << "N = " <<N<<endl;
        // for(int i=0; i<sliced.size(); i++) cout<<sliced[i]<<"  ";
        // cout<<"\nslice size = "<<sliced.size()<<endl;
        // cout<<"Line size = "<<line_size<<endl;
        // cout<<"dim = "<<dim<<endl;


    }
    else{
        cout<<"The .in file cannot be opened.\n";
        exit(202);
    }

}

void extract(
    double* x, double* v, double* m,
    double* sliced,const unsigned& N, const unsigned& dim
){
    // copy from sliced to x,y,m
    int line_size = 2*dim+1;
    for (int i=0; i<N; ++i){
        for(int j=0; j<dim; ++j){
            int pos_x = i*line_size+j+1;
            int pos_v = i*line_size+dim+j+1;
            x[i*dim+j] = sliced[pos_x];
            v[i*dim+j] = sliced[pos_v];
            // cout<<sliced[pos_x]<<" : "<<x[i*dim+j]
            //     <<" ? "<<sliced[pos_v]<<" : "<< v[i*dim+j]<<endl;
        }
        int pos_m = i*line_size;
        m[i] = sliced[pos_m];
        // cout<<sliced[pos_m]<<" : "<<m[i]<<endl;
    }
}
