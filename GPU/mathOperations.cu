#include <iostream>
#include <cmath>
#include <vector>
#include "mathOperations.cuh"
#include "read_bk.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

/*
// Compute gradient CPU
template <class T>
void gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad){
    int a , b;
    for (int e = 0 ; e < numEdges; e++){
        a = mEdge[e].start;
        b = mEdge[e].end;
        grad[e] = w[e] * (x[b] - x[a]);
    }
}

// Compute divergence
template<typename T>
void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){
    int nbhd_vertices, sign, edge;
    T temp;
    for (int v =0 ; v<numNodes; v++){
        nbhd_vertices = mVert[v].nbhdVert.size();
        temp = 0;
        for (int j = 0; j< nbhd_vertices ; j++){
            sign = mVert[v].sign[j];
            edge = mVert[v].nbhdEdges[j];
            temp += sign*w[edge]*p[edge];
        }
        divg[v] = temp;
    }
}*/

//COMPUTE GRADIENT GPU
 template <class T> __device__ void gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad){
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

// COMPUTE DIVERGENCE GPU
template <class T> __device__ void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){

    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int v = tnum_x + tnum_y + tnum_z; 

    int nbhd_vertices, sign, edge;
    T temp = 0;
    if (v< numNodes){
        /*nbhd_vertices = mVert[v].nbhdVert.size();
        for (int j = 0; j< nbhd_vertices ; j++){
            sign = mVert[v].sign[j];
            edge = mVert[v].nbhdEdges[j];
            temp += sign*w[edge]*p[edge];
        }*/
        divg[v] = temp;
    }
}

// Compute L1 norm
/*template <class T>
void compute_L1 (T *grad_x, T &x_norm, int num_vertex){
    // Compute L1 norm
    x_norm = 0;
    for (size_t i = 0; i < num_vertex; i++){
        x_norm += abs(grad_x[i]);
    }
}*/

//IMPORTANT:::: PLEASE READ THIS

//https://stackoverflow.com/questions/12400477/retaining-dot-product-on-gpgpu-using-cublas-routine


// Compute the absolute sum of the array using cublas library


/*
cublasHandle_t handle;
cublasCreate(&handle);
cublasDasum(handle, num_vertex, grad_x, 1, &x_norm);
*/

// Compute root mean square
/*template <class T>
void compute_RMS (T *gap_vec, T &gap, int num_vertex){
    gap = 0;
    // Compute L2 norm of gap
    for (size_t i = 0; i < num_vertex; i++){
        gap += gap_vec[i] * gap_vec[i];
    }
    gap = (T) sqrt(gap/num_vertex);
}*/

// Compute the L2 norm of a vector .. use   for single precision and cublasDnrm2 for double
/*
cublasHandle_t handle;
cublasCreate(&handle);*/
/*
gap = 0;
cublasDnrm2(handle, num_vertex, grad_x, 1, &gap);
gap = (T) sqrt(gap/num_vertex);
*/

// Compute scalar product
/*template <class T>
void compute_scalar_product (T *x, T *f, T &xf, int num_vertex){
    // Compute scalar product
    xf = 0;
    for (size_t i = 0; i < num_vertex; i++){
        xf += x[i]*f[i];
    }
}*/

// Compute the dot product of two vectors

/*
cublasHandle_t handle;
cublasCreate(&handle);
cublasDdot(handle, num_vertex, x, 1, f, 1, &xf);
*/

template __device__ void gradient_calculate <float>(float*, float*, edge*, int, float*);
template __device__ void gradient_calculate <double>(double*, double*, edge*, int, double*);
template __device__ void divergence_calculate <float>(float*, float*, vert*, int, float*);
template __device__ void divergence_calculate <double>(double*, double*, vert*, int, double*);
/*
template void compute_L1 <float> (float*, float&, int);
template void compute_L1 <double> (double*, double&, int);
template void compute_RMS <float> (float*, float&, int);
template void compute_RMS <double> (double*, double&, int);
template void compute_scalar_product <float> (float*, float*, float&, int);
template void compute_scalar_product <double> (double*, double*, double&, int);
*/
