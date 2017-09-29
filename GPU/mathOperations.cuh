#ifndef __MATHOPERATIONS_H__
#define __MATHOPERATIONS_H__

#include <iostream>
#include <math.h>
#include "read_bk.h"

template <class T> 
__global__ void h_gradient_calculate(T *w, T *x, int* d_start_edge, int* d_end_edge , int numEdges, T *grad);

template <class T>
 __global__ void h_divergence_calculate(T* w, T* p, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, int numNodes, T* divg);

#endif
