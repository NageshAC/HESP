#include <cuda.h>
#include <iostream>
using namespace std;

void cudasafe(int error, string message, string file, int line) {
        if (error != cudaSuccess) {
                cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
                exit(-1);
        }
}


int main(int argc, char ** argv) {
        int deviceCount; 

        cudasafe(cudaGetDeviceCount(&deviceCount), "GetDeviceCount", __FILE__, __LINE__); 

        cout<<"Number of CUDA devices: "<<deviceCount<<endl; 

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        int cuda_v;

        cudasafe(cudaRuntimeGetVersion(&cuda_v), "Get Runtime Version",__FILE__, __LINE__);

        cudasafe(cudaGetDeviceProperties(&deviceProp, dev), "Get Device Properties", __FILE__, __LINE__);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                cout<<"No CUDA GPU has been detected\n";
                return -1;
            } else if (deviceCount == 1) {
                cout<<"There is 1 device supporting CUDA\n";
            } else {
                cout<<"There are "<<deviceCount<<" devices supporting CUDA\n";
            }
        }

                cout<<"For device #"<<dev<<"\n"; 
                cout<<"Device name:                "<<deviceProp.name<<endl;
                cout<<"CUDA Version:               "<<cuda_v<<endl;
                cout<<"Major revision number:      "<<deviceProp.major<<endl;
                cout<<"Minor revision Number:      "<<deviceProp.minor<<endl;  
                cout<<"Total Global Memory:        "<<deviceProp.totalGlobalMem<<endl; 
                cout<<"Total shared mem per block: "<<deviceProp.sharedMemPerBlock<<endl; 
                cout<<"Total const mem size:       "<<deviceProp.totalConstMem<<endl;
                cout<<"Warp size:                  "<<deviceProp.warpSize<<endl;
                cout<<"Maximum block dimensions:   "<<deviceProp.maxThreadsDim[0]
                <<" x "<<deviceProp.maxThreadsDim[1]<<" x "<<deviceProp.maxThreadsDim[2]<<""<<endl;

                cout<<"Maximum grid dimensions:    "<<deviceProp.maxGridSize[0]<<" x "
                <<deviceProp.maxGridSize[1]<<" x "<<deviceProp.maxGridSize[2]<<""<<endl;
                cout<<"Clock Rate:                 "<<deviceProp.clockRate<<endl; 
                cout<<"Number of muliprocessors:   "<<deviceProp.multiProcessorCount<<endl;  

   } 

    return 0;


}
