#ifndef __MATHOPERATIONS_H__
#define __MATHOPERATIONS_H__

#include <iostream>
#include <math.h>
#include "read_bk.h"

template <class T> __global__
void h_gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad);

template <class T> __global__ 
void h_divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg);

#endif
