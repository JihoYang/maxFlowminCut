#include <iostream>
#include <cmath>
#include "mathOperations.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

using namespace std;

template <class T> 
__global__ void h_gradient_calculate(T *w, T *x,int* d_start_edge, int* d_end_edge , int numEdges, T *grad){
    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int e = tnum_x + tnum_y + tnum_z; 

    int a , b;
    if ( e< numEdges){
        a = d_start_edge[e];
        b = d_end_edge[e];
        grad[e] = w[e] * (x[b] - x[a]);
    }
}

// COMPUTE DIVERGENCE GPU - Global
template <class T>
 __global__ void h_divergence_calculate(T* w, T* p, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, int numNodes, T* divg){

    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int v = tnum_x + tnum_y + tnum_z; 

    int nbhd_vertices, sign, edge;
    T temp = 0;
    if (v< numNodes){
        nbhd_vertices = d_nbhd_size[v];
        for (int j = 0; j< nbhd_vertices ; j++){
            sign = d_nbhd_sign[d_nbhd_start[v] + j];
            edge = d_nbhd_edges[d_nbhd_start[v] + j];
            temp += sign*w[edge]*p[edge];
        }
        divg[v] = temp;
    }
}

template __global__ void h_gradient_calculate <float>(float*, float*, int*, int* , int, float*);
template __global__ void h_gradient_calculate <double>(double*, double*, int*, int*, int, double*);
template __global__ void h_divergence_calculate <float>(float*, float*, int*,int*, int*,int*, int, float*);
template __global__ void h_divergence_calculate <double>(double*, double*, int*,int*, int*, int*, int, double*);
