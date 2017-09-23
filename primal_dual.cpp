#include <iostream>
#include <math.h>
#include <vector>
#include "primal_dual.h"

using namespace std;

// Update U
template <class T>
void updateU(T *u, T *tau, T *div_y, T *y, T *f, T *f_y, int num_vertex){
     T u_new;
     for (size_t i = 0; i < num_vertex; i++){
		///////////////////////////////////////////

		// Compute divergence of y (output = div_y)

		///////////////////////////////////////////
		// Compute new u
        u_new = u[i] + tau[i] * (div_y[i] - f[i]);
		// Compute 2u_(t+1) - u_t for y update
        f_y[i] = 2*u_new - u[i];    
		// Clamping
        if (u_new < 0)
            u_new = 0;
         else if (u_new > 1)
            u_new = 1;
         else if (u_new > 0 && u_new < 0.5)
             u_new = 0;
         else if (u_new >= 0.5 && u_new < 1)
             u_new = 1;
		// Update u
        u[i] = u_new;
     }
}

// Update Y
template <class T>
void updateY(T *y, T *sigma, T *grad_f_y, int num_edge){
	T y_new;
    for (size_t i = 0; i < num_edge; i++){
		/////////////////////////////////////////////////////////

		// Compute gradient of 2u_(t+1) - u_t (output = grad_f_y)

		/////////////////////////////////////////////////////////
		// Compute new y
         y_new = y[i] + sigma[i] * grad_f_y[i];
		// Clamping
        if (y_new < - 1)
        	y_new = -1;
        else if (y_new > 1)
        	y_new = 1;
        else if (y_new > -1 && y_new < 0.5)
            y_new = -1;
        else if (y_new < 1 && y_new >= 0.5)
           y_new = 1;
		// Update y
        y[i] = y_new;
    }   
}      

// Compute time steps
template <class T>
void compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, int num_vertex, int num_edge){
	// Get the smaller value for loop
	int lower = min(num_vertex, num_edge);
    int upper = max(num_vertex, num_edge);
    for (size_t i = 0; i < lower; i++){
    	// Tau
		T sum = (T)0;
        for (size_t j = 0; j < num_vertex; j++){
			// I need data structure to get the right index for w_u[i]
			//w_u[];
        	sum += pow(abs(w_u[i][j]), alpha);
        }
        tau[i] = 1 / (phi * sum);
        // Sigma
		sigma[i] = phi / pow(abs(w_u[i]), 2 - alpha);
	}
}

// Compute L1 norm
template <class T>
void compute_L1 (T *grad_u, T &u_norm, int num_vertex){
	// Compute L1 norm
	u_norm = 0;
	for (size_t i = 0; i < num_vertex; i++){
		u_norm += abs(grad_u[i]);
	}
}

// Compute L2 norm
template <class T>
void compute_L2 (T *gap_vec, T &gap, int num_vertex){
	gap = 0;
	// Compute L2 norm of gap
	for (size_t i = 0; i < num_vertex; i++){
		gap += gap_vec[i] * gap_vec[i];
	}
	gap = (T)sqrt(gap);
}

// Compute scalar product
template <class T>
void compute_scalar_product (T *u, T *f, T &uf, int num_vertex){
	// Compute scalar product
	uf = 0;
	for (size_t i = 0; i < num_vertex; i++){
		uf += u[i]*f[i];
	}
}

// Compare 0 and div_y - f
template <class T>
void get_max (T *div_y, T *f, T *max_vec, int num_vertex){
	// Get max value
	for (size_t i = 0; i < num_vertex; i++){
		max_vec[i] = max(0, div_y[i] - f[i]);
	}
}

// Compute gap
template <class T>
void compute_gap(T *u, T *f, T *div_y, T &gap, int num_vertex, int num_edge){
	// Allocate memory
	vector<T *> grad_u(num_edge);
	vector<T *> max_vec(num_vertex);
	vector<T *> gap_vec(num_vertex);

	////////////////////////

	// Compute gradient of u
	
	////////////////////////

	// Parameters
	T uf, u_norm;
	// Compute scalar product
	compute_scalar_product(u, f, uf, num_vertex);
	// Compute L1 norm of gradient of u
	compute_L1(grad_u, u_norm, num_vertex);
	// Compare 0 and div_y - f
	get_max(div_y, f, max_vec, num_vertex);
	// Compute gap
	for (size_t i = 0; i < num_vertex; i++){
		gap_vec[i] = uf + u_norm + max_vec[i];
	}
	// Compute L2 norm of gap
	compute_L2(gap_vec, gap, num_vertex);
}
