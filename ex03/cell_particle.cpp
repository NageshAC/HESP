#pragma once
#include<iostream>
#include<cmath>

using namespace std;

struct particle{
    double *x=NULL, *v=NULL, *f=NULL;

    particle() = default;

    particle(double* tempx, double* tempv, double* tempf){
        this->x = tempx;
        this->v = tempv;
        this->f = tempf;
    }

    particle(const particle& other){
        particle* p = new particle;
        this->x = other.x;
        this->v = other.v;
        this->f = other.f;
    }

    particle& operator=(const particle& other){
        this->x = other.x;
        this->v = other.v;
        this->f = other.f;
        return *this;
    }

    double distance(const particle& other, const int dim){
        double dis = 0;
        for(int i=0;i<dim;i++){
            dis += pow(this->x[i]-other.x[i],2); 
        }
        return sqrt(dis);
    }
};

typedef particle* particleptr;

class cell{
    private: 
    particleptr part[10];
    int n = 0;
    int count = 0;

    public:
    // cell(const int n){
    //     this->part = new particle[n];
    //     this->n = n;
    // }

    ~cell(){
        delete[] this->part;
    }

    void addParticle(particleptr x){
        part[count] = x;
        count++;
    }

    void delParticle(particleptr x){
        int flag = 0;
        for(int i = 0; i<count; i++){
            if (part[i] == x){
                part[i] = part[count-1];
                flag = 1;
                count--;
            }
        }
        if(flag == 0){
            cout<<"Error: No particle found to delete";
        }
    }


};