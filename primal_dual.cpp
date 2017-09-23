#include <iostream>
#include <math.h>
#include <vector>
#include "primal_dual.h"
#include "read_bk.h"

using namespace std;

// Compute gradient
template <class T> 
void gradient_calculate(T *w, T *x, edge *mEdge , int numEdges, T *grad){
     int a , b;
     for (int e = 0 ; e < numEdges; e++){
         a = mEdge[e].start;
         b = mEdge[e].end;
         grad[e] = w[e] * (x[a] - x[b]);
     }
}

// Compute divergence 
template<typename T> 
void divergence_calculate(T* w, T* p, vert *mVert, int numNodes, T* divg){
     int ndhd_vertices, sign, edge;
     for (int v =0 ; v<numNodes; v++){
         ndhd_vertices = mVert[v].nbhdVert.size();
         for (int j = 0; j< ndhd_vertices ; j++){
             sign = mVert[v].sign[j];
             edge = mVert[v].nbhdEdges[j];
             divg[v] += sign*w[edge]*p[edge];
         }

     }
}

// Update X
template <class T>
void updateX(T *w, vert *mVert, T *x, T *tau, T *div_y, T *y, T *f, T *f_y, int num_vertex){
     T x_new;
     for (size_t i = 0; i < num_vertex; i++){
		// Compute divergence of y (output = div_y)
		divergence_calculate(w, y, mVert, num_vertex, div_y);
		// Compute new u
        x_new = x[i] + tau[i] * (div_y[i] - f[i]);
		// Compute 2u_(t+1) - u_t for y update
        f_y[i] = 2*x_new - x[i];    
		// Clamping
        if (x_new < 0)
            x_new = 0;
         else if (x_new > 1)
            x_new = 1;
         else if (x_new > 0 && x_new < 0.5)
             x_new = 0;
         else if (x_new >= 0.5 && x_new < 1)
             x_new = 1;
		// Update u
        x[i] = x_new;
     }
}

// Update Y
template <class T>
void updateY(T *w, T *x, edge *mEdge, T *y, T *sigma, T *x_diff, T *grad_x_diff, int num_edge){
	T y_new;
    for (size_t i = 0; i < num_edge; i++){
		// Compute gradient of 2u_(t+1) - u_t (output = grad_f_y)
		gradient_calculate(w, x, mEdge , num_edge, grad_x_diff);
		// Compute new y
         y_new = y[i] + sigma[i] * grad_x_diff[i];
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
void compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, vert *mVert, int num_vertex, int num_edge){
	// Size of neighbouring vertices j for vertex i
	int size_nbhd;
	// Compute tau
    for (size_t i = 0; i < num_vertex; i++){
		T sum = (T)0;
		size_nbhd = mVert[i].nbhdVert.size();
        for (size_t j = 0; j < size_nbhd; j++){
			sum += pow(abs(w_u[mVert[i].nbhdVert[j]]), alpha);
        }
        tau[i] = 1 / (phi * sum);
	}
	// Compute sigma
	for (size_t i = 0; i < num_edge; i++){
		sigma[i] = phi / pow(abs(w_u[i]), 2 - alpha);
	}
}

// Compute L1 norm
template <class T>
void compute_L1 (T *grad_x, T &x_norm, int num_vertex){
	// Compute L1 norm
	x_norm = 0;
	for (size_t i = 0; i < num_vertex; i++){
		x_norm += abs(grad_x[i]);
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
void compute_scalar_product (T *x, T *f, T &xf, int num_vertex){
	// Compute scalar product
	xf = 0;
	for (size_t i = 0; i < num_vertex; i++){
		xf += x[i]*f[i];
	}
}

// Compare 0 and div_y - f
template <class T>
void get_max (T *div_y, T *f, T *max_vec, int num_vertex){
	// Get max value
	for (size_t i = 0; i < num_vertex; i++){
		max_vec[i] = max((T)0, div_y[i] - f[i]);
	}
}

// Compute gap
template <class T>
void compute_gap(T *w, edge *mEdge, T *x, T *f, T *div_y, T &gap, int num_vertex, int num_edge){
	// Allocate memory
	T *grad_x = new T[num_edge];
	T *max_vec = new T[num_vertex];
	T *gap_vec = new T[num_vertex];
	// Compute gradient of u
	gradient_calculate(w, x, mEdge, num_edge, grad_x);
	// Parameters
	T xf, x_norm;
	// Compute scalar product
	compute_scalar_product(x, f, xf, num_vertex);
	// Compute L1 norm of gradient of u
	compute_L1(grad_x, x_norm, num_vertex);
	// Compare 0 and div_y - f
	get_max<T>(div_y, f, max_vec, num_vertex);
	// Compute gap
	for (size_t i = 0; i < num_vertex; i++){
		gap_vec[i] = xf + x_norm + max_vec[i];
	}
	// Compute L2 norm of gap
	compute_L2(gap_vec, gap, num_vertex);
	// Free memory
	delete []grad_x;
	delete []max_vec;
	delete []gap_vec;
}

template void compute_dt<float>(float*, float*, float*, float, float, vert*, int, int);
template void compute_dt<double>(double*, double*, double*, double, double, vert*, int, int);
template void compute_gap<float>(float*, edge*, float*, float*, float*, float&, int, int);
template void compute_gap<double>(double*, edge*, double*, double*, double*, double&, int, int);
template void updateX<float>(float*, vert*, float*, float*, float*, float*, float*, float*, int);
template void updateX<double>(double*, vert*, double*, double*, double*, double*, double*, double*, int);
template void updateY<float>(float*, float*, edge*, float*, float*, float*, float*, int);
template void updateY<double>(double*, double*, edge*, double*, double*, double*, double*, int);


