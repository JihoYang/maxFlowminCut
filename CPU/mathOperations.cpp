#include <iostream>
#include <cmath>
#include <vector>
#include "mathOperations.h"
#include "read_bk.h"

using namespace std;

// Compute gradient
template <class T>
void gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad){
    int a , b;
    for (int e = 0 ; e < numEdges; e++){
        a = mEdge[e].start;
        b = mEdge[e].end;
        grad[e] = w[e] * (x[b] - x[a]);
        //cout << "a="<<a<<", b="<<b<<", x_b="<<x[b]<<", x_a="<<x[a]<<endl;  
    }
}

// Compute divergence
template<typename T>
void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){
    int nbhd_vertices, sign, edge;
    T temp;
    for (int i =0 ; i<numNodes; i++){
        nbhd_vertices = mVert[i].nbhdVert.size();
        temp = 0;
        for (int j = 0; j< nbhd_vertices ; j++){
            sign = mVert[i].sign[j];
            edge = mVert[i].nbhdEdges[j];
            temp += sign*w[edge]*p[edge];
        }
        divg[i] = temp;
    }
}

// Compute L1 norm
template <class T>
void compute_L1 (T *grad_x, T &x_norm, int num_vertex){
    // Compute L1 norm
    x_norm = 0;
    for (size_t i = 0; i < num_vertex; i++){
        x_norm += abs(grad_x[i]);
    }
}

// Compute root mean square
template <class T>
void compute_RMS (T *gap_vec, T &gap, int num_vertex){
    gap = 0;
    // Compute L2 norm of gap
    for (size_t i = 0; i < num_vertex; i++){
        gap += gap_vec[i] * gap_vec[i];
    }
    gap = (T) sqrt(gap/num_vertex);
}

// Compute scalar product
template <class T>
void compute_scalar_product (T *x, T *f, T &xf, int num_vertex){
    // Compute scalar product
    xf = 0;
    for (size_t i = 0; i < num_vertex; i++){
        xf += x[i]*f[i];
    }
}

template<class T>
void roundVector(T* x, int num_elem)
{
    for(int i = 0; i < num_elem; i++)
    {
        if(x[i] <(T)0.5) x[i] = (T)0;
        else x[i] = (T)1;
    } 
}


template void gradient_calculate <float>(float*, float*, edge*, int, float*);
template void gradient_calculate <double>(double*, double*, edge*, int, double*);
template void divergence_calculate <float>(float*, float*, vert*, int, float*);
template void divergence_calculate <double>(double*, double*, vert*, int, double*);
template void compute_L1 <float> (float*, float&, int);
template void compute_L1 <double> (double*, double&, int);
template void compute_RMS <float> (float*, float&, int);
template void compute_RMS <double> (double*, double&, int);
template void compute_scalar_product <float> (float*, float*, float&, int);
template void compute_scalar_product <double> (double*, double*, double&, int);
template void roundVector <double> (double*, int);
template void roundVector <float> (float*, int);