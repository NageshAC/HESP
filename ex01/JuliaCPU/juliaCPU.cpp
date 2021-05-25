#include <cmath>
#include <iostream>
#include "lodepng.h"
#include <complex>
#include <chrono>

using namespace std;
const int bit_color = 8;
const int h = 2048, w = h;
const double xmax = 2.0, xmin = -2.0;
// const double cx = 0, cy = 0.8;
const double xresolution = (xmax-xmin)/w;
const double yresolution = (xmax-xmin)/h;
const int threshold = 10;
unsigned low_itr = 300, high_itr = 0;

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
// unsigned setvalue( int n, double cx,double cy){
//     double zx = xmin + (n%w)*xresolution , zy = xmax - (n/w)*yresolution;
//     // cout<<"x: "<<x<<"  y: "<<y<<endl;
//     unsigned i = 0;
//     while(sqrt(pow(zx,2) + pow(zy,2))<10 && i<200){
//         zx = zx*zx-zy*zy + cx;
//         zy = 2*zx*zy + cy;
//         ++i;
//     }
//     if (i<low_itr) low_itr = i;
//     if (i>high_itr) high_itr = i;
//     // cout<< "i: " <<i<<endl;
//     return (i);
// }
unsigned setvalue( int n, double cx,double cy){
    complex<double> z (xmin + (n%w)*xresolution , xmin + (n/w)*yresolution);
    // double x = xmin + (n%w)*xresolution , y = xmax - (n/w)*yresolution;
    // cout<<"x: "<<x<<"  y: "<<y<<endl;
    // cout<<z<<"\n";
    complex<double> c (cx,cy);
    unsigned i = 0;
    while(abs(z)<threshold){
        z = z*z + c;
        ++i;
    }
    if (i<low_itr) low_itr = i;
    if (i>high_itr) high_itr = i;
    // cout<< "i: " <<i<<endl;
    return (i);
}
void setcolor(unsigned& image){
    // image = (image -low_itr)*pow(2,bit_color)/(high_itr-low_itr);
    // if (image < 10) image = 0;
    // else image = (image - low_itr)* 10;
    // image *= 10;
    image = (image - low_itr)* 10;
}
void initBG(unsigned char* img){
    for (int i = 0; i<w*h*3; i+=3) {
        img[i+1] = 0; 
        img[i+2] = 0;
    }
}
void createImg(double cx, double cy, string filename = "juliaCPUx.png"){
    unsigned char* img = new unsigned char [w*h*3];
    unsigned* image = new unsigned [w*h];
    initBG(img);
    for (int i = 0; i<w*h; i++) image[i] = setvalue(i,cx,cy);
    // cout<<"high: "<<high_itr<<"  low: "<<low_itr<<endl;
    for (int i = 0; i<w*h; i++) setcolor(image[i]);
    for (int i = 0; i<w*h*3; i+=3) img[i] = static_cast<unsigned char>(image[i/3]);
    delete[] image;
    lodepng::encode(filename, img, w, h, LCT_RGB, bit_color);
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