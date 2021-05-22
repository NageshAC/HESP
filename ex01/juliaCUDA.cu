#include <cmath>
#include <iostream>
#include "lodepng.h"
#include <chrono>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;
typedef long long ll;

const int bit_color = 8;
const int h = 512, w = h;
const double xmax = 2.0, xmin = -2.0;
// // const double cx = 0, cy = 0.8;
const double xresolution = (xmax-xmin)/w;
// const double yresolution = (xmax-xmin)/h;
// const int threshold = pow(10,2);
// unsigned low_itr = 300, high_itr = 0;

class Timer {
    // using declarations for improving readability of the code
    using hrc =  chrono::high_resolution_clock;
    using time_point =  chrono::time_point<hrc>;
    using micro_sec =  chrono::microseconds;
    using milli_sec = chrono::milliseconds;
    using sec = chrono::seconds;
    typedef long long  ll;

    private:
        time_point start_time_;
        time_point stop_time_;
        ll duration_ {0};

    public:
        Timer() = default;
        ~Timer() = default;

        void start(){
            start_time_ = hrc::now();
        }

        void stop(){
            stop_time_ = hrc::now();
        }

        ll duration_us(){
            auto s =  chrono::time_point_cast<micro_sec>(start_time_).time_since_epoch().count();
            auto e =  chrono::time_point_cast<micro_sec>(stop_time_).time_since_epoch().count();
            duration_ = e - s; 
            return duration_;
        }

        ll duration_ms(){
            auto s =  chrono::time_point_cast<milli_sec>(start_time_).time_since_epoch().count();
            auto e =  chrono::time_point_cast<milli_sec>(stop_time_).time_since_epoch().count();
            duration_ = e - s; 
            return duration_;
        }

        ll duration_s(){
            auto s =  chrono::time_point_cast<sec>(start_time_).time_since_epoch().count();
            auto e =  chrono::time_point_cast<sec>(stop_time_).time_since_epoch().count();
            duration_ = e - s; 
            return duration_;
        }
        
};

void initBG(unsigned char* img){
    for (int i = 0; i<w*h*3; i+=3) {
        img[i+1] = 0; 
        img[i+2] = 0;
    }
}
thrust::complex<double> initR(int idx){
    thrust::complex<double> z (xmin + (idx%w)*xresolution , xmin + (idx/w)*xresolution);
    return z;
}
void cudasafe(int error, string message="(---)", string file = "(-this file-)", int line = -1) {
    if (error != cudaSuccess) {
            cout<<stderr<< " CUDA Error: "<<message<<" : "<<error<<". In "<<file<<" line "<<line<<endl; 
            exit(-1);
    }
}
__global__ void setv(int* image, const double cx,const double cy, thrust::complex<double>* z){
    int idx = blockIdx.x * 32*32 + threadIdx.y * 32 + threadIdx.x;
    thrust::complex<double> c (cx,cy);
    unsigned i = 0;
    while(abs(z[idx]) <10 && i<300){
        z[idx] = z[idx]*z[idx] + c ;
        ++i;
    }

    
    image[idx] = i*10;
}
void createImg(double cx, double cy, string filename = "juliaCUDAx.png"){
    unsigned char* img = new unsigned char [w*h*3];
    int* image = new int [w*h];
    vector<thrust::complex<double>> z(w*h);
    thrust::complex<double>* d_z;
    int* d_img;
    dim3 blocks(32,32);
    dim3 grids(w*h/pow(32,2));

    cudasafe(cudaMalloc(&d_img, w*h*sizeof(int)), "Mem Allo", __FILE__,__LINE__);
    cudasafe(cudaMalloc(&d_z, w*h*sizeof(thrust::complex<double>)), "Mem Allo", __FILE__,__LINE__);

    for(int i=0; i<w*h; i++)z[i] = initR(i);

    cudasafe(cudaMemcpy(d_z,&z[0], w*h*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice), "Mem Allo", __FILE__,__LINE__);

    setv <<< grids, blocks >>> (d_img,cx,cy,d_z);

    initBG(img);
    
    cudasafe(cudaDeviceSynchronize(),"Sync", __FILE__,__LINE__);
    cudasafe(cudaMemcpy(image, d_img, w*h*sizeof(int), cudaMemcpyDeviceToHost), "Mem Trans", __FILE__,__LINE__);

    // for(int i=0; i<w*h; i++)cout<<image[i]<<"  ";

    for (int i = 0; i<w*h*3; i+=3) {
        img[i] = static_cast<unsigned char>(image[i/3]);   
    }

    lodepng::encode(filename, img, w, h, LCT_RGB, bit_color);

    cudasafe(cudaFree(d_img),"Mem free", __FILE__,__LINE__);
    delete[] image;
    delete[] img;
}

int main(int argc, char* argv[]){
    Timer t;
    t.start();
    createImg(stof(argv[1]), stof(argv[2]), argv[3]);
    t.stop();
    cout<<"\n\n Total time taken: "<< t.duration_s()<<"  s = "<<t.duration_ms()<<"  ms"<<endl;
    return 0;
}