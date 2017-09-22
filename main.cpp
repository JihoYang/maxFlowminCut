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

//Primal-dual algorithm
void updateX(float *u, float *tau, float *grady, float *y, float *f, float *f_y, int num_vertex){
	float u_new;
	for (size_t i = 0; i < num_vertex; i++){
		u_new = u[i] + tau[i] * (grady[i] - f[i]);
		f_y[i] = 2*u_new - u[i];	
		if (u_new < 0)
			u_new = 0;
		else if (u_new > 1)
			u_new = 1;
	    else if (u_new > 0 && u_new < 0.5)
			u_new = 0;
		else if (u_new >= 0.5 && u_new < 1)
			u_new = 1;
		u[i] = u_new;
	}
}

void updateY(float *y, float *sigma, float *grad_f_y, int num_edge){
	float y_new;
	for (size_t i = 0; i < num_edge; i++){
		y_new = y[i] + sigma[i] * grad_f_y[i];
		if (y_new < - 1) 
			y_new = -1;
		else if (y_new > 1)
			y_new = 1;
		else if (y_new > -1 && y_new < 0.5)
			y_new = -1;
		else if (y_new < 1 && y_new >= 0.5)
			y_new = 1;
		y[i] = y_new;
	}
}

void compute_dt(float *tau, float *sigma, float **w, float alpha, float phi, int num_vertex, int num_edge){
	float lower = min(num_vertex, num_edge);
	float upper = max(num_vertex, num_edge);
	for (size_t i = 0; i < lower; i++){
		// Tau
		float sum = 0;
		for (size_t j = 0; j < num_vertex; j++){
			sum += pow(abs(w[i][j]), alpha);
		}
		tau[i] = 1 / (phi * sum);
		// Sigma

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

	for (int i=0; i < 4; i++){
		for (int j=0; j < 4; j++){
			cout << "w[" << i << "]" << "[" << j << "]" << " = " << w[i][j] << endl;
		}
	}

	return 0;

	delete []u;
	delete []f;
	delete []w;
}
