#ifndef __PRIMAL_DUAL_CUH__
#define __PRIMAL_DUAL_CUH__

#include <iostream>
#include <math.h>
#include "read_bk.h"

// GPU
template <class T>
__global__ void updateX(T *x, T *y, T *w, T *f, T *x_diff, T *div_y, vert *mVert, T *tau, int num_vertex);

template <class T>
__global__ void updateY(T *x_diff, T *y, T *w, edge *mEdge, T *sigma, int num_edge);

template <class T> 
__global__ void d_compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, vert *mVert, int num_vertex, int num_edge);

template <class T> 
__global__ void max_vec_computation (T *div_y, T *f, T *max_vec, T &max_val, int num_vertex);

#endif 			  
