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
        grad[e] = w[e] * (x[a] - x[b]);
    }
}

// Compute divergence
template<typename T>
void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){
    int ndhd_vertices, sign, edge;
    for (int v =0 ; v<numNodes; v++){
        ndhd_vertices = mVert[v].nbhdVert.size();
        for (int j = 0; j< ndhd_vertices ; j++){
            sign = mVert[v].sign[j];
            edge = mVert[v].nbhdEdges[j];
            divg[v] += sign*w[edge]*p[edge];
        }
        
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
