#include <iostream>
#include <math.h>
using namespace std;

void compute_f(float *f, float *u, float **w, int &num_vertex){
	for (size_t j = 1; j < num_vertex-1; ++j){
		float sum1 = 0;
		float sum2 = 0;
		f[j] = -w[0][j] + w[j][num_vertex-1];
		for (size_t i = 1; i < j; ++i){
			if (i != j){
				sum1 += w[i][j];
				sum2 += w[j][i];
			}
		}
		f[j] = f[j] - sum1 + sum2;
	}
}

void compute_w(float **w, int &num_vertex){
	for (size_t i = 0; i < num_vertex; ++i){
		for (size_t j = 0; j < num_vertex; ++j){
			w[i][j] = (w[i][j] + w[j][i])/2;
			w[j][i] = 0;
		}
	}
}


int main(int argc, char **argv){

	int num_vertex = 5;
	int num_edge   = 5;
	// Memory allocation
	float *u = new float[num_vertex];
	float *f = new float[num_vertex];
	f[0] = 0; f[num_vertex-1] = 0;
	float **w = new float*[num_vertex];
	for (size_t i = 0; i < num_vertex; ++i){
		 w[i] = new float[num_vertex];
		for (size_t j = 0; j < num_vertex; ++j){
			w[i][j] = 0;
		}
	}
	// Graph
	u[0] = 40;
	u[1] = 0;
	u[2] = 0;
	u[3] = -100;
	u[4] = -40;
	w[0][1] = 20;
	w[0][2] = 20;
	w[2][1] = 30;
	w[1][3] = 40;
	w[2][3] = 10;
	// Compute f
	compute_f(f, u, w, num_vertex);
	// Compute omega
	compute_w(w, num_vertex);
	return 0;

	delete []u;
	delete []f;
	delete []w;
}
