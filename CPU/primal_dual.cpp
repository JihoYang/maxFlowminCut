#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "primal_dual.h"
#include "mathOperations.h"
#include "read_bk.h"

using namespace std;

// Compute time steps
template <class T>
void compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, vert *mVert, int num_vertex, int num_edge){
    // Size of neighbouring vertices j for vertex i
    int size_nbhd;
    // Compute tau
    for (size_t i = 0; i < num_vertex; i++){
        T sum = (T)0;
        size_nbhd = mVert[i].nbhdVert.size();
		if (size_nbhd > 0) {
			for (size_t j = 0; j < size_nbhd; j++){
				sum += pow(abs(w_u[mVert[i].nbhdEdges[j]]), alpha);
			}
			tau[i] = (T)1 / ((T)phi * (T)sum);
		}	
		else
		{
			tau[i] = 0; 
		} 
    }
    // Compute sigma
    for (size_t i = 0; i < num_edge; i++){
        sigma[i] = (T)phi / (T)pow(abs(w_u[i]), 2 - alpha);
    }
}

// Compare 0 and div_y - f
template <class T>
void get_max (T *div_y, T *f, T *max_vec, T &sum, int num_vertex){
	sum = (T) 0;
	// Get max value and sum the results
    for (size_t i = 0; i < num_vertex; i++){
        max_vec[i] = max( (T) 0, div_y[i] - f[i] );
		sum += max_vec[i];
    }
}

// Update X
template <class T>
void updateX(T *w, vert *mVert, T *x, T *tau, T *div_y, T *y, T *f, T *x_diff, int num_vertex){
     T x_new;
	 for (size_t i = 0; i < num_vertex; i++)
	 {
		// Compute divergence of y (output = div_y)
		divergence_calculate<T>(w, y, mVert, num_vertex, div_y);
		// Compute new u
        x_new = x[i] + tau[i] * (div_y[i] - f[i]);
		// Compute 2u_(t+1) - u_t for y update
        x_diff[i] = 2*x_new - x[i];    
		// Clamping
        if (x_new < 0)
            x_new = 0;
        else if (x_new > 1)
            x_new = 1;
        // Update u
        x[i] = x_new;
     }
}

// Update Y
template <class T>
void updateY(T *w, T *x, edge *mEdge, T *y, T *sigma, T *x_diff, T *grad_x_diff, int num_edge){
	T y_new;
	// Compute gradient of 2u_(t+1) - u_t (output = grad_x_diff)
	gradient_calculate<T>(w, x_diff, mEdge , num_edge, grad_x_diff);

	/*for (int i=0; i<num_edge; i++){
		cout <<"Gradient _"<<i<<" is "<< grad_x_diff[i] << endl;
	}*/
    
    for (size_t i = 0; i < num_edge; i++){
		// Compute new y
		 y_new = y[i] + sigma[i] * grad_x_diff[i];
		// Clamping
        if (y_new < - 1)
        	y_new = -1;
        else if (y_new > 1)
        	y_new = 1;
        // Update y
        y[i] = y_new;
    }   
}      

// Compute gap
template <class T>
void compute_gap(T *w, edge *mEdge, vert *mVert, T *x, T *y, T *f, T &gap, T &x_norm, T &xf, int num_vertex, int num_edge){
	// Allocate memory
	T *grad_x = new T[num_edge];
	T *max_vec = new T[num_vertex];
	T *div_y = new T[num_vertex];
	T *gap_vec = new T[num_vertex];
	T max_val;
	// Compute gradient of u
	gradient_calculate<T>(w, x, mEdge, num_edge, grad_x);
	// Compute scalar product
	compute_scalar_product<T>(x, f, xf, num_vertex);
	// Compute L1 norm of gradient of u
	compute_L1<T>(grad_x, x_norm, num_edge);
	divergence_calculate<T>(w, y, mVert, num_vertex, div_y);
	// Compare 0 and div_y - f
	get_max<T>(div_y, f, max_vec, max_val, num_vertex);
	//cout << " Xf = " << xf << " x_norm = " << x_norm << " max_val = " << max_val << endl;
	// Compute gap
	gap = (xf + x_norm + max_val) / num_edge;
	// Free memory
	delete []grad_x;
	delete []max_vec;
	delete []gap_vec;
}

template void compute_dt<float>(float*, float*, float*, float, float, vert*, int, int);
template void compute_dt<double>(double*, double*, double*, double, double, vert*, int, int);
template void compute_gap<float>(float*, edge*, vert*, float*, float*, float*, float&, float&, float&, int, int);
template void compute_gap<double>(double*, edge*, vert*, double*, double*, double*, double&, double&, double&, int, int);
template void updateX<float>(float*, vert*, float*, float*, float*, float*, float*, float*, int);
template void updateX<double>(double*, vert*, double*, double*, double*, double*, double*, double*, int);
template void updateY<float>(float*, float*, edge*, float*, float*, float*, float*, int);
template void updateY<double>(double*, double*, edge*, double*, double*, double*, double*, int);
