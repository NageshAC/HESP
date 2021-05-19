#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<cstring>
using namespace std;

template<typename T>
T* init(T* f, unsigned N){
    f = new T [N];
    for(int i=0; i<N; i++) f[i] = 0;
    return f;
}

void readParam(
    const string& paramFileName,
    string& part_input_file,string &part_out_name_base,string &vtk_out_name_base,
    double& timeStep,double& timeEnd,double& epsilon,double& sigma,
    unsigned& part_out_freq,unsigned& vtk_out_freq,unsigned& cl_wg_1dsize

){
    fstream paramFile;
    paramFile.open(paramFileName, ios::in);
    if(paramFile.is_open()){
        // cout<<"Parameter file is opened.\n";
        string out1, out2;
        
        while(!paramFile.eof()){
            paramFile >> out1;
            paramFile >> out2;
            // cout<<out1<<"-"<<out2<<endl;
            if(out1 == "part_input_file") part_input_file = out2;
            if(out1 == "timestep_length") stringstream(out2) >> timeStep;
            if(out1 == "time_end") stringstream(out2) >> timeEnd;
            if(out1 == "epsilon") stringstream(out2) >> epsilon;
            if(out1 == "sigma") stringstream(out2) >> sigma;
            if(out1 == "part_out_freq") stringstream(out2) >> part_out_freq;
            if(out1 == "part_out_name_base") part_out_name_base = out2;
            if(out1 == "vtk_out_freq") stringstream(out2) >> vtk_out_freq;
            if(out1 == "vtk_out_name_base") vtk_out_name_base = out2;
            if(out1 == "cl_workgroup_1dsize") stringstream(out2) >> cl_wg_1dsize;
        }
        cout<<"Done reading paramerters file.\n\n";
    }
    else{
        cout<<"The input file cannot be opened.\n";
        exit(202);
    }
    paramFile.close();
}

void writeParam(
    const string& part_input_file,const string& part_out_name_base,const string& vtk_out_name_base,
    double& timeStep,double& timeEnd,double& epsilon,double& sigma,
    unsigned& part_out_freq,unsigned& vtk_out_freq,unsigned& cl_wg_1dsize

){
    cout<<"\n--- printing parameters ---\n"
        <<"\npart_input_file "<< part_input_file
        <<"\ntimestep_length "<<timeStep
        <<"\ntime_end "<<timeEnd
        <<"\nepsilon "<< epsilon
        <<"\nsigma "<<sigma
        <<"\npart_out_freq "<<part_out_freq
        <<"\npart_out_name_base "<<part_out_name_base
        <<"\nvtk_out_freq "<<vtk_out_freq
        <<"\nvtk_out_name_base "<<vtk_out_name_base
        <<"\ncl_workgroup_1dsize "<<cl_wg_1dsize
        <<"\n------------------------------\n";

}
void readInput(
        const string& infile, double *x, double *v, 
        double *m, unsigned& N, unsigned& dim){
    fstream ifile;
    ifile.open(infile,ios::in);
    if(ifile.is_open()){
        cout<<"Input file is opened.\n";

        string out;
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

        for(int i=0; i<sliced.size(); i++) cout<<sliced[i]<<"  ";

        cout<<"\nslice size = "<<sliced.size()<<endl;
        cout<<"Line size = "<<line_size<<endl;
        cout<<"dim = "<<dim<<endl;

        m = init(m,N); x = init(x,(N*dim)); v = init(v,(N*dim));
        // init(m,N); init(x,(N*dim)); init(v,(N*dim));


        for(int i=0; i<N; i++){
            for (int j =0; j<dim; j++){
                int x_slice = i*line_size+j;
                int v_slice = i*line_size+dim+j;
                int idx = i*dim+j;
                // stringstream(sliced[i*line_size+j]) >> x[i*dim+j];
                // stringstream(sliced[i*line_size+dim+j]) >> v[i*dim+j];
                // cout<<endl<<i*line_size+j<<" : x "<<i*dim+j
                //     <<"  -->  "<<sliced[i*line_size+j]<<" : "<<x[i*dim+j];
                // cout<<endl<<i*line_size+dim+j<<" : v "<<i*dim+j
                //     <<"  -->  "<<sliced[i*line_size+dim+j] <<" : "<<v[i*dim+j];

                stringstream(sliced[x_slice]) >> x[idx];
                stringstream(sliced[v_slice]) >> v[idx];
                cout<<endl<<x_slice<<" : x "<<idx
                    <<"  -->  "<<sliced[x_slice]<<" : "<<x[idx];
                cout<<endl<<v_slice<<" : v "<<idx
                    <<"  -->  "<<sliced[v_slice] <<" : "<<v[idx];

                // cout << "\n\n";
                // for(unsigned itr = 0; itr < (N*dim); ++itr)
                //     cout << x[itr] << "\t";
                // cout << "\n\n";
                // for(unsigned itr = 0; itr < (N*dim); ++itr)
                //     cout << v[itr] << "\t";
                // cout << "\n\n";
            }
            stringstream(sliced[(i+1)*line_size-1]) >> m[i];
        }

        ifile.close();
        cout<<"\nInput file closed.\n";
    }
    else{
        cout<<"The input file cannot be opened.\n";
        exit(202);
    }
    
}

void writeInput(
        string infile, double *x, double *v, 
        double *m, unsigned N, unsigned dim){

    cout<<"\n--- Printing input file ---\n\n"<<N<<endl;
    for(int j=0; j<N; j++){
        for(int i=0; i<dim; i++) cout<<x[j*(dim)+i]<<" ";
        for(int i=0; i<dim; i++) cout<<v[j*(dim)+i]<<" ";
        cout<<m[j]<<endl;
    }
}


int main(){

    //reading input parameters
    string paramFileName = "attract.par";
    string part_input_file, part_out_name_base, vtk_out_name_base;
    double timeStep, timeEnd, epsilon, sigma;
    unsigned part_out_freq, vtk_out_freq, cl_wg_1dsize;
    readParam(
        paramFileName,
        part_input_file, part_out_name_base, vtk_out_name_base,
        timeStep, timeEnd, epsilon, sigma,
        part_out_freq, vtk_out_freq, cl_wg_1dsize
    );
    // writeParam(
    //     part_input_file, part_out_name_base, vtk_out_name_base,
    //     timeStep, timeEnd, epsilon, sigma,
    //     part_out_freq, vtk_out_freq, cl_wg_1dsize
    // );

    // initialising the setup;
    unsigned N , dim;
    double *x, *v, *m;
    // double *f, *fold;
    readInput(part_input_file,x,v,m,N,dim); 
    // init(f , N * dim); init(fold , N * dim);
    // writeInput(part_input_file,x,v,m,N,dim);


    // delete[] x,v,m;
    // delete[] f,fold;
    return 0;
}


