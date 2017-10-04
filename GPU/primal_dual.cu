#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "primal_dual.cuh"
#include "mathOperations.cuh"
#include <cublas_v2.h>

using namespace std;

// COMPUTE DIVERGENCE (GPU)
template <class T>
 __device__ void divergence_calculate(T* w, T* p, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, int v, T &divg){

    int nbhd_vertices, sign, edge;
    T temp = 0;
    nbhd_vertices = d_nbhd_size[v];
    for (int j = 0; j< nbhd_vertices ; j++){
        sign = d_nbhd_sign[d_nbhd_start[v] + j];
        edge = d_nbhd_edges[d_nbhd_start[v] + j];
        temp += sign*w[edge]*p[edge];
    }
    divg = temp;
}

// Update X (GPU)
template <class T> 
__global__ void updateX(T *x, T *y, T *w, T *f, T *x_diff, T *div_y, int* d_nbhd_size, int* d_nbhd_start, int* d_nbhd_sign, int* d_nbhd_edges, T *tau, int num_vertex){
	// Get coordinates - 3D coordinate implemented for sake of generality (perhaps we could play around with different block configurations)
	int x_thread = threadIdx.x + blockDim.x * blockIdx.x;
	int y_thread = threadIdx.y + blockDim.y * blockIdx.y;
	int z_thread = threadIdx.z + blockDim.z * blockIdx.z;
	// Get indices
	int idx = x_thread + y_thread + z_thread;
	// Temporary values
	T x_new;
	// Compute new u
	if (idx < num_vertex){
		// Compute divergence of y 
		divergence_calculate <T> (w, y, d_nbhd_size, d_nbhd_start, d_nbhd_sign, d_nbhd_edges, idx, div_y[idx]);
		// Compute u
		x_new = x[idx] + tau[idx] * (div_y[idx] - f[idx]);
		if (x_new < 0)
			x_new = 0;
		else if (x_new > 1)
			x_new = 1;
		// Compute x_diff
		x_diff[idx] = 2*x_new - x[idx];
		// Update X	
		x[idx] = x_new;
	}
}

//COMPUTE GRADIENT (GPU)
template <class T> 
__device__ void gradient_calculate(T *w, T *x,int* d_start_edge, int* d_end_edge , int e, T &grad){
    int a , b;
    a = d_start_edge[e];
    b = d_end_edge[e];
    grad = w[e] * (x[b] - x[a]);

}

// Update Y (GPU)
template <class T>
__global__ void updateY(T *x_diff, T *y, T *w, int* d_start_edge, int* d_end_edge, T *sigma, int num_edge){
	// Get coordinates
	int x_thread = threadIdx.x + blockDim.x * blockIdx.x;
	int y_thread = threadIdx.y + blockDim.y * blockIdx.y;
	int z_thread = threadIdx.z + blockDim.z * blockIdx.z;
	// Get indices
	int idx = x_thread + y_thread + z_thread;
	if (idx < num_edge){
		// Temporary values
		T y_new, grad_x_diff;
		// Compute gradient of x_diff
		gradient_calculate <T> (w, x_diff, d_start_edge, d_end_edge, idx, grad_x_diff);
		// Compute new y
		y_new = y[idx] + sigma[idx] * grad_x_diff;
		// Clamping
		if (y_new < -1)
			y_new = -1;
		else if (y_new > 1)
			y_new = 1;
		// Update y
		y[idx] = y_new;
	}
}

// Compute dt (GPU)
template <class T> 
 __global__ void d_compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, int *d_nbhd_size, int*d_nbhd_edges ,int* d_nbhd_start, int num_vertex, int num_edge){
    // Size of neighbouring vertices j for vertex i
 	int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int i = tnum_x + tnum_y + tnum_z; 

    int size_nbhd = d_nbhd_size[i]; 
	int start_nbhd = d_nbhd_start[i];

	// If there are no neighbours set tau to be zero .... no need as we are setting d_tau to 0
    if (i < num_vertex && size_nbhd > 0){

		T sum = (T)0;
		for (size_t j = 0; j < size_nbhd; j++){ 
				sum += pow(abs(w_u[d_nbhd_edges[start_nbhd + j]]), alpha); 
			}
		tau[i] = (T)1 / ((T)phi * (T)sum);
    }
    // Compute sigma
    if (i < num_edge){
        sigma[i] = (T)phi / (pow((T)abs(w_u[i]), (T) 2 - (T) alpha)*2);
    }
}

template <class T> 
__global__ void max_vec_computation (T *div_y, T *f, T *max_vec, int num_vertex){
	int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
	int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
	int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
	int i = tnum_x + tnum_y + tnum_z; 

	// Get max value and sum the results
	if (i < num_vertex){
	max_vec[i] = max( (T) 0, div_y[i] - f[i] );
	}
}

template __global__ void updateX <float> (float*, float*, float*, float*, float*, float*, int* , int* , int* , int* , float*, int);
template __global__ void updateX <double> (double*, double*, double*, double*, double*, double*, int* , int* , int* , int* , double*, int );

template __global__ void updateY <float> (float*, float*, float*, int* , int* , float *, int);
template __global__ void updateY <double> (double*, double*, double*, int* , int* , double*, int);

template __global__ void d_compute_dt<float>(float*, float*, float*, float, float, int* , int*, int*, int, int);
template __global__ void d_compute_dt<double>(double*, double*, double*, double, double, int*, int*, int*, int, int);

template __global__ void max_vec_computation <float > (float*, float*, float*, int );
template __global__ void max_vec_computation <double> (double*, double*, double*, int );

template __device__ void gradient_calculate <float>(float*, float*, int*, int*, int, float&);
template __device__ void gradient_calculate <double>(double*, double*, int*, int*, int, double&);

template __device__ void divergence_calculate <float>(float*, float*, int*, int*, int*, int*, int, float&);
template __device__ void divergence_calculate <double>(double*, double*, int*, int*, int*, int*, int, double&);
