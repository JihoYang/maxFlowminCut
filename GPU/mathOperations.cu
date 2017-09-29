#include <iostream>
#include <cmath>
#include "mathOperations.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

using namespace std;

template <class T> __global__
void h_gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad){
    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int e = tnum_x + tnum_y + tnum_z; 

    int a , b;
    if (e< numEdges){
        a = mEdge[e].start;
        b = mEdge[e].end;
        grad[e] = w[e] * (x[b] - x[a]);
    }
}

// COMPUTE DIVERGENCE GPU - Global
template <class T> __global__ void h_divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){

    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int v = tnum_x + tnum_y + tnum_z; 

    int nbhd_vertices, sign, edge;
    T temp = 0;
    if (v< numNodes){
        nbhd_vertices = mVert[v].nbhdSize;
        for (int j = 0; j< nbhd_vertices ; j++){
            sign = mVert[v].sign[j];
            edge = mVert[v].nbhdEdges[j];
            temp += sign*w[edge]*p[edge];
        }
        divg[v] = temp;
    }
}

template __global__ void h_gradient_calculate <float>(float*, float*, edge*, int, float*);
template __global__ void h_gradient_calculate <double>(double*, double*, edge*, int, double*);
template __global__ void h_divergence_calculate <float>(float*, float*, vert*, int, float*);
template __global__ void h_divergence_calculate <double>(double*, double*, vert*, int, double*);
