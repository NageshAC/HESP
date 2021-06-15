#pragma once
#include<iostream>
#include<cmath>
using namespace std;


struct particle{
    unsigned id, cell_id;
    double *x=NULL, *v=NULL, *f=NULL;
    particle *next = NULL, *prev = NULL;

    particle() = default;

    particle(unsigned& tempid, double* tempx, double* tempv, double* tempf){
        this->id = tempid;
        this->x = tempx;
        this->v = tempv;
        this->f = tempf;
        // cout<<x[0]<<"\t"<<x[1]<<"\t"<<x[2]<<"\n";
    }

    particle(const particle& other){
        this->id = other.id;
        this->cell_id = other.cell_id;
        this->x = other.x;
        this->v = other.v;
        this->f = other.f;
    }

    particle& operator=(const particle& other){
        this->id = other.id;
        this->x = other.x;
        this->v = other.v;
        this->f = other.f;
        return *this;
    }

    double distance(const particle* other, const int dim){
        double dis = 0;
        for(int i=0;i<dim;i++){
            dis += pow(this->x[i]-other->x[i],2); 
        }
        return sqrt(dis);
    }

    void del(){
        particle* temp = this->prev;
        temp->next = this->next;
        temp = this->next;
        temp->prev = this->prev;
        this->prev=NULL;
        this->next=NULL;
    }

    void calCell_id(
        double& x_min, double& del_x,
        double& y_min, double& del_y,
        double& z_min, double& del_z,
        unsigned& x_n, unsigned& y_n
    ){
        int i_x = (x[0]-x_min)/del_x;
        int i_y = (x[1]-y_min)/del_y;
        int i_z = (x[2]-z_min)/del_z;
        cell_id = i_z*x_n*y_n + i_y*x_n + i_x;
    }
};
typedef particle* particleptr;


struct cell{
    particle *first, *last;

    cell() = default;

    void addParticle(particle* x){
        if(last){
            last->next = x;
            x->prev = last;
            last = x;
        }
        else{
            first = x;
            last = x;
        }
    }

    cell& operator=(const cell& other){
        this->first = other.first;
        this->last = other.last;
        return *this;
    }
};

