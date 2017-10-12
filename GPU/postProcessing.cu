#include <iostream>
#include <stdio.h>
#include "postProcessing.cuh"

// Round the solutions
template <class T>
__global__ void round_solution(T *d_x, int numNodes){
	// Get index
	int x_thread = threadIdx.x;
	int idx = x_thread;
	// Round the solution
	if (idx < numNodes){
		if (d_x[idx] > 0.5)
			d_x[idx] = 1;
		else
			d_x[idx] = 0;
	}
}

//template void export_result <float> (const char*, float*, int);
template __global__ void round_solution <float> (float*, int);
template __global__ void round_solution <double> (double*, int);

