#ifndef __PRIMAL_DUAL_CUH__
#define __PRIMAL_DUAL_CUH__

#include <iostream>
#include <math.h>
#include "read_bk.h"

template <class T> 
__device__ void gradient_calculate(T *w, T *x, int* d_start_edge, int* d_end_edge, int e, T &grad);

template <class T>
 __device__ void divergence_calculate(T* w, T* p, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, int v, T& divg);

template <class T> 
__global__ void updateX(T *x, T *y, T *w, T *f, T *x_diff, T *div_y, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, T *tau, int num_vertex);

template <class T>
__global__ void updateY(T *x_diff, T *y, T *w, int* d_start_edge, int* d_end_edge, T *sigma, int num_edge);

template <class T> 
__global__ void d_compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, int *nbhd_size, int*nbhd_edges ,int* d_nbhd_start,int num_vertex, int num_edge);

template <class T> 
__global__ void max_vec_computation (T *div_y, T *f, T *max_vec, int num_vertex);


#endif 			  
