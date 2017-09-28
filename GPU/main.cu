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
		// Compute gap
		d_compute_gap <float> (d_g->w, d_g->E, d_x, d_g->f, d_div_y, gap, x_norm, xf, numNodes, numEdges, grid, block);
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
    return 0;
}
