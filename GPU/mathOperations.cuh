#ifndef __MATHOPERATIONS_H__
#define __MATHOPERATIONS_H__

#include <iostream>
#include <math.h>
#include "read_bk.h"

template <class T> __device__
void gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad);

template <class T> __global__
void h_gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad);

template <class T> __device__ 
void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg);

template <class T> __global__ 
void h_divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg);

/*template <class T>
void compute_L1 (T *grad_x, T &x_norm, int num_vertex);

template <class T>
void compute_RMS (T *gap_vec, T &gap, int num_vertex);

template <class T>
void compute_scalar_product (T *x, T *f, T &xf, int num_vertex);
*/
#endif
