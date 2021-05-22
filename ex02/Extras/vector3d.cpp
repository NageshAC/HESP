#pragma once
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

template<typename T>
class vector3d{

    private:
        T x,y,z;
    
    public:
       __device__ __host__ __inline__  vector3d();
       __device__ __host__ __inline__  vector3d(T a, T b = 0, T c = 0) {
            this->x = a;
            this->y = b;
            this->z = c;
        };

       __device__ __host__ __inline__ void setx(T a){this->x = a;}
       __device__ __host__ __inline__ void sety(T a){this->y = a;}
        __device__ __host__ __inline__ void setz(T a){this->z = a;}

        __device__ __host__ __inline__ T getx(){return this->x;}
        __device__ __host__ __inline__ T gety(){return this->y;}
        __device__ __host__ __inline__ T getz(){return this->z;}

        __device__ __host__ __inline__ vector3d<T> get(){
            // T* g[3];
            return *this;
        }

        __device__ __host__ __inline__ friend vector3d<T> operator+ (const vector3d<T>& v1, const vector3d<T>& v2){
            return vector3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        }

        template <typename P>
        __device__ __host__ __inline__ friend vector3d<T> operator+(const vector3d<T>& v1, const P& temp){
            return vector3d(v1.x + temp, v1.y + temp, v1.z + temp);
        }

        __device__ __host__ __inline__ friend vector3d<T> operator- (const vector3d<T>& v1, const vector3d<T>& v2){
            return vector3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        }

        template <typename P>
        __device__ __host__ __inline__ friend vector3d<T> operator-(const vector3d<T>& v1, const P& temp){
            return vector3d(v1.x - temp, v1.y - temp, v1.z - temp);
        }

        __device__ __host__ __inline__ friend vector3d<T> operator* (const vector3d<T>& v1, const vector3d<T>& v2){
            return vector3d(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
        }

        template <typename P>
        __device__ __host__ __inline__ friend vector3d<T> operator*(const vector3d<T>& v1, const P& temp){
            return vector3d(v1.x * temp, v1.y * temp, v1.z * temp);
        }
        
        __device__ __host__ __inline__ friend vector3d<T> operator/ (const vector3d<T>& v1, const vector3d<T>& v2){
            return vector3d(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
        }

        __device__ __host__ __inline__ friend ostream& operator<<(ostream& out, const vector3d<T> v){
            out <<"\nx = "<<v.x<<"\n"
                <<"y = "<<v.y<<"\n"
                <<"z = "<<v.z<<"\n";
            return out;
        }

};