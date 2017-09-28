//////////////////////////////////////////////////////////////////////////
//																		//
//		GPU accelerated max flow min cut graph problem solver			//
//																		//
//		Written by: Apoorva Gupta										//
//					Jorge Salazar										//
//					Jiho Yang											//
//			numEdges															//
//		Final update: 26/09/2017										//
//																		//
//////////////////////////////////////////////////////////////////////////
	
// TODO: the code diverges - check for the correctness of primal-dual algorithm and grad/div computation

#include <iostream>
#include <vector>
#include <time.h>
#include "read_bk.h"
#include "primal_dual.cuh"
#include "mathOperations.cuh"
#include "postProcessing.h"
#include <string.h>
#include <cublas_v2.h>

# define T float

using namespace std;

int main(int argc, char **argv)
{
    if (argc <= 1)
	{
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
    }
	// Start time
	clock_t tStart = clock();
	// Parameters
	float alpha = 1;
	float rho = 1;
	float gap = 1;
	float eps = 1E-6;
	int it  = 0;
	int iter_max = 500;
	float xf;
	float x_norm;
	float max_flow;
	T max_val;
	const char *method = "PD_CPU";
	
	// Import bk file    
	read_bk<float> *g = new read_bk<float>(argv[1]); 	
	int numNodes  = g->nNodes;
	int numEdges = g->nEdges;
	float *f = g->f;
	float *w = g->w;
	float b = g->b;
	vert* mVert = g->V;
	edge* mEdge = g->E;

	// Allocate memory on host
	float *x = new float[numNodes];
	float *y = new float[numEdges];
	float *div_y = new float[numNodes];
	float *x_diff = new float[numNodes];
	float *grad_x_diff = new float[numEdges];
	float *tau = new float[numNodes];
	float *sigma = new float[numEdges];

	// Names of all the cuda_arrays	
	read_bk<float> *d_g;
 	float * d_x, *d_y, *d_div_y, *d_x_diff, *d_grad_x_diff, *d_tau, *d_sigma;
	
	// Allocate memory on cuda	
	cudaMalloc((void**)&d_g, sizeof(read_bk<float>));
	cudaMalloc((void**)&d_x, numNodes*sizeof(float));
	cudaMalloc((void**)&d_y, numEdges*sizeof(float));
	cudaMalloc((void**)&d_div_y, numNodes*sizeof(float));
	cudaMalloc((void**)&d_x_diff, numNodes*sizeof(float));
	cudaMalloc((void**)&d_grad_x_diff, numEdges*sizeof(float));
	cudaMalloc((void**)&d_tau, numNodes*sizeof(float));
	cudaMalloc((void**)&d_sigma, numEdges*sizeof(float));

	// Initialise cuda memories
	cudaMemcpy(d_g, g, sizeof(read_bk<float>), cudaMemcpyHostToDevice);	
	cudaMemset(d_x , 0, numNodes*sizeof(float));
	cudaMemset(d_y , 0, numEdges*sizeof(float));
	cudaMemset(d_div_y , 0, numNodes*sizeof(float));
	cudaMemset(d_x_diff , 0, numNodes*sizeof(float));
	cudaMemset(d_grad_x_diff , 0, numEdges*sizeof(float));
	cudaMemset(d_tau , 0, numNodes*sizeof(float));
	cudaMemset(d_sigma , 0, numEdges*sizeof(float));

	cublasHandle_t handle;
	cublasCreate(&handle);

	T *d_grad_x, *d_max_vec, *d_gap_vec;
	cudaMalloc((void**)&d_grad_x, numEdges*sizeof(float));
	cudaMalloc((void**)&d_max_vec, numNodes*sizeof(float));
	cudaMalloc((void**)&d_gap_vec, numNodes*sizeof(float));
	
	dim3 block = dim3(1024,1,1);
	int grid_x = ((max(numNodes, numEdges) + block.x - 1)/block.x);
	int grid_y = 1;
	int grid_z = 1;
	dim3 grid = dim3(grid_x, grid_y, grid_z );

	// Pre-compute time steps
	d_compute_dt <float> <<<grid, block>>> (d_tau, d_sigma, d_g->w, alpha, rho, d_g->V, numNodes, numEdges);
	// Iteration
	cout << "------------------- Time loop started -------------------"  << endl;
	while (it < iter_max && gap > eps){
		// Update X
		//updateX <float> (w, mVert, x, tau, div_y, y, f, x_diff, numNodes);

		//********************************************** Compute gap
		// Update X
		updateX <float> <<< grid, block >>> (d_x, d_y, d_g->w, d_g->f, d_x_diff, d_div_y, d_g->V, d_tau, numNodes);
		// Update Y
		updateY <float> <<<grid, block >>> (d_x_diff, d_y, d_g->w, d_g->E, d_sigma, numEdges);
		// Update divergence of Y
		h_divergence_calculate(d_g->w, d_y, d_g->V, numNodes, d_div_y);
		// Compute gradient of u
		h_gradient_calculate <T> <<<grid, block>>>(d_g->w, d_x, d_g->E, numEdges, d_grad_x);
		// Compute L1 norm of gradient of u
		cublasSasum(handle, numNodes, d_grad_x, 1, &x_norm);
		// Compute scalar product
		cublasSdot(handle, numNodes, d_x, 1, d_g->f, 1, &xf);	
		// Compare 0 and div_y - f
		max_vec_computation <T> <<<grid, block >>> (d_div_y, d_g->f, d_max_vec, numNodes);
		cublasSasum(handle, numNodes, d_max_vec, 1, &max_val);
		//cout << " Xf = " << xf << " x_norm = " << x_norm << " max_val = " << max_val << endl;
		// Compute gap
		gap = (xf + x_norm + max_val) / numEdges;
		
		//******************************************* Compute Gap end
		cout << "Iteration = " << it << endl << endl;
		cout << "Gap = " << gap << endl << endl;
		it = it + 1;
		// Update Y for next iteration
		//updateY <float> (w, x, mEdge, y, sigma, x_diff, grad_x_diff, numEdges);
	}
	// End time
	clock_t tEnd = clock();
	// Compute max flow
	max_flow = xf + x_norm + b;

	cout << "Max flow = " << max_flow << endl << endl;


	// Program exit messages
	if (it == iter_max) cout << "ERROR: Maximum number of iterations reached" << endl << endl;
	cout << "------------------- End of program -------------------"  << endl << endl;
	cout << "Execution Time = " << (double)1000*(tEnd - tStart)/CLOCKS_PER_SEC << " ms" << endl << endl;
	//Export results
	//export_result <float> (method, x, numNodes);
	// Free memory    
	delete g;
	
	/*
	delete []x;
	delete []y;
	delete []x_diff;
	delete []div_y;
	delete []grad_x_diff;
	delete []tau;
	delete []sigma;	
	*/

	cudaFree(d_grad_x);
	cudaFree(d_max_vec);
	cudaFree(d_gap_vec);
    return 0;
}
