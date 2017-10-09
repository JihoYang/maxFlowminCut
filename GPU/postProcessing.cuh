#ifndef __POSTPROCESSING_CUH__
#define __POSTPROCESSING_CUH__

#include <iostream>                                                                                                                                                                                     
#include <stdio.h>

using namespace std;

template <class T>
__global__ void round_solution(T *d_x, int numNodes);

//template <class T>
//void export_result(const char *method, T *x, int numNodes);

#endif
