#ifndef __PRIMAL_DUAL_H__
#define __PRIMAL_DUAL_H__

#include <iostream>
#include <math.h>
#include "read_bk.h"

// GPU
__global__ void primal_dual(float *x, float *y, float *w, float *f, float *div_y, vert *mVert, edge *mEdge, float *tau, float *sigma, int num_vertex, int num_edge);
__device__ void updateY(float *x, float *y, float &x_diff, float *w, edge *mEdge, float *sigma, int num_edge);

template <class T> __global__
void d_compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, vert *mVert, int num_vertex, int num_edge);

template <class T> __global__
void max_vec_computation (T *div_y, T *f, T *max_vec, int num_vertex);

template <class T>
void updateX (T *w, vert *mVert, T *x, T *tau, T *div_y, T *, T *f, T *x_diff, int num_vertex);

template <class T>
void updateY (T *w, T *x, edge *mEdge, T *y, T *sigma, T *x_diff, T *grad_x_diff, int num_edge);

#endif 			  
