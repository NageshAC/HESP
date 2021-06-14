#pragma once
#include<iostream>
#include<cmath>

typedef particle* particleptr;

using namespace std;

struct particle{
    unsigned id;
    double *x=NULL, *v=NULL, *f=NULL;
    particle *next = NULL, *prev = NULL;
    cell* head;

    particle() = default;

    particle(unsigned& tempid, double* tempx, double* tempv, double* tempf){
        this->id = tempid;
        this->x = tempx;
        this->v = tempv;
        this->f = tempf;
    }

    particle(const particle& other){
        this->id = other.id;
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

    double distance(const particle& other, const int dim){
        double dis = 0;
        for(int i=0;i<dim;i++){
            dis += pow(this->x[i]-other.x[i],2); 
        }
        return sqrt(dis);
    }

    void join(cell *c){
        c->addParticle(this, c);
    }

    void del(){
        particleptr temp = this->prev;
        temp->next = this->next;
        temp = this->next;
        temp->prev = this->prev;
        this->prev=NULL;
        this->next=NULL;
    }
};

class cell{
    private: 
    particleptr even=NULL, odd=NULL, evenlast=NULL, oddlast=NULL;
    unsigned id;
    public:
    cell() = default;

    void addParticle(particleptr x, cell* c){
        if(x->id%2 == 0){ //even
            if(evenlast==NULL){
                even = x;
                evenlast = x;
                x->head=c;
            }
            else{
                evenlast->next=x;
                x->prev=evenlast;
                evenlast=x;
                x->head=c;
            }
        }
        
        if(x->id%2 == 1){ //odd
            if(oddlast==NULL){
                odd = x;
                oddlast = x;
                x->head=c;
            }
            else{
                oddlast->next=x;
                x->prev=oddlast;
                oddlast=x;
                x->head=c;
            }
        }
    }

    void delParticle(particleptr x){
        particleptr temp = x->prev;
        temp->next = x->next;
        temp = x->next;
        temp->prev = x->prev;
        x->prev=NULL;
        x->next=NULL;
    }
    
    void set_cell_id(unsigned& id){
        this->id = id;
    }
    unsigned get_cell_id(){
        return this->id;
    }
    particleptr get_even_ptr(){
        return even;
    }
    particleptr get_odd_ptr(){
        return odd;
    }

};