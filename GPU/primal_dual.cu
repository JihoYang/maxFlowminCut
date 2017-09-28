#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "primal_dual.cuh"
#include "mathOperations.cuh"
#include <cublas_v2.h>

using namespace std;

// Update X (GPU)
template <class T> 
__global__ void updateX(T *x, T *y, T *w, T *f, T *x_diff, T *div_y, vert *mVert, T *tau, int num_vertex){
	// Get coordinates - 3D coordinate implemented for sake of generality (perhaps we could play around with different block configurations)
	int x_thread = threadIdx.x + blockDim.x * blockIdx.x;
	//int y_thread = threadIdx.y + blockDim.y * blockIdx.y;
	//int z_thread = threadIdx.z + blockDim.z * blockIdx.z;
	// Get indices
	//int idx = x_thread + (size_t)w*y_thread + (size_t)w*h*z_thread;
	int idx = x_thread;
	// Temporary values
	float x_new;
	// Compute divergence of y 
	divergence_calculate <T> (w, y, mVert, num_vertex, div_y);
	// Compute new u
	if (idx < num_vertex){
		// Compute u
		x_new = x[idx] + tau[idx] * (div_y[idx] - f[idx]);
		// Compute x_diff
		x_diff[idx] = x_new - x[idx];
		if (x_new < 0)
			x_new = 0;
		else if (x_new > 1)
			x_new = 1;
		// Update X	
		x[idx] = x_new;
	}
}

// Update Y (GPU)
template <class T>
__global__ void updateY(T *x_diff, T *y, T *w, edge *mEdge, T *sigma, int num_edge){
	// Get coordinates
	int x_thread = threadIdx.x + blockDim.x * blockIdx.x;
	//int y_thread = threadIdx.y + blockDim.y * blockIdx.y;
	//int z_thread = threadIdx.z + blockDim.z * blockIdx.z;
	// Get indices
	//int idx = x_thread + (size_t)w*y_thread + (size_t)w*h*z_thread;
	int idx = x_thread;
	// Temporary values
	float y_new, grad_x_diff;
	// Compute gradient of x_diff
	gradient_calculate <T> (w, x_diff, mEdge, num_edge, grad_x_diff);
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

template <class T> 
__global__ void d_compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, vert *mVert, int num_vertex, int num_edge){
    // Size of neighbouring vertices j for vertex i
    int size_nbhd;
    // Compute tau
    int tnum_x = threadIdx.x + blockIdx.x*blockDim.x;
    int tnum_y = threadIdx.y + blockIdx.y*blockDim.y;
    int tnum_z = threadIdx.z + blockIdx.z*blockDim.z;
    int i = tnum_x + tnum_y + tnum_z; 
	// If there are no neighbours set tau to be zero
    if (i < num_vertex){
        T sum = (T)0;
        size_nbhd = mVert[i].nbhdSize;
		if (size_nbhd == 0){ 
			tau[i] = 0;
		}
		else if (size_nbhd != 0) {
			for (size_t j = 0; j < size_nbhd; j++){
				sum += pow(abs(w_u[mVert[i].nbhdEdges[j]]), alpha);
			}
			tau[i] = (T)1 / ((T)phi * (T)sum);
		}
    }
    // Compute sigma
    if (i<num_edge){
        sigma[i] = (T)phi / pow((T)abs(w_u[i]), (T) 2 - (T) alpha);
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

template __global__ void updateX <float> (float *x, float *y, float *w, float *f, float *x_diff, float *div_y, vert *mVert, float *tau, int num_vertex);
template __global__ void updateX <double> (double *x, double *y, double *w, double *f, double *x_diff, double *div_y, vert *mVert, double *tau, int num_vertex);

template __global__ void updateY <float> (float *x_diff, float *y, float *w, edge *mEdge, float *sigma, int num_edge);
template __global__ void updateY <double> (double *x_diff, double *y, double *w, edge *mEdge, double *sigma, int num_edge);

template __global__ void d_compute_dt<float>(float*, float*, float*, float, float, vert*, int, int);
template __global__ void d_compute_dt<double>(double*, double*, double*, double, double, vert*, int, int);

template __global__ void max_vec_computation (float *div_y, float *f, float *max_vec, int num_vertex);
template __global__ void max_vec_computation (double *div_y, double *f, double *max_vec, int num_vertex);
