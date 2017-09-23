#include <iostream>
#include <stdio.h>
#include "postProcessing.h"

// Export results
template <class T>
void export_result(const char *method, T *x, int numNodes){
	// Create a file
	char FileName[80];
	sprintf(FileName, "%s_displacement.csv", method);
	// Write results to this file
	FILE *f = fopen(FileName, "wb");
	for (int i = 0; i < numNodes; i++){
		fprintf(f, " %i %i\n", i, (int)x[i]);
	}
	fclose(f);
}

template void export_result <float> (const char*, float*, int);
