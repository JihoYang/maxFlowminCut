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
		if (d_x[idx] >= 0.5)
			d_x[idx] = 1;
		else
			d_x[idx] = 0;
	}
}

/*
// Export results
template <class T>
void export_result(const char *method, T *x, int numNodes){
	// Create a file
	char FileName[80];
	sprintf(FileName, "%s.csv", method);
	// Write results to this file
	FILE *f = fopen(FileName, "wb");
	for (int i = 0; i < numNodes; i++){
		fprintf(f, " %i %i\n", i, (int)x[i]);
	}
	fclose(f);
}
*/

//template void export_result <float> (const char*, float*, int);
template __global__ void round_solution <float> (float*, int);
template __global__ void round_solution <double> (double*, int);

