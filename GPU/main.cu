//////////////////////////////////////////////////////////////////////////
//																		//
//		GPU accelerated max flow min cut graph problem solver			//
//																		//
//		Written by: Apoorva Gupta										//
//					Jorge Salazar										//
//					Jiho Yang											//
//																		//
//		Final update: 26/09/2017										//
//																		//
//////////////////////////////////////////////////////////////////////////
	

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
	int iter_max = 100;
	float xf;
	float x_norm;
	float max_flow;
	T max_val;
	//const char *method = "PD_CPU";
	
	// Import bk file    
	read_bk<float> *g = new read_bk<float>(argv[1]); 	
	int numNodes  = g->nNodes;
	int numEdges = g->nEdges;
	
	// Allocating and initializing f and w on the device
	float *f = g->f;
	float *w = g->w;
	T *d_f , *d_w;
	cudaMalloc((void**)&d_f , numNodes*sizeof(T));
	cudaMalloc((void**)&d_w , numEdges*sizeof(T));
	cudaMemcpy(d_f , f, numNodes*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w , w, numEdges*sizeof(T), cudaMemcpyHostToDevice);

	// Allocating and initializing the start and end of edge on the device
	edge* mEdge = g->E;

	int *start_edge = new int[numEdges];
	int *end_edge = new int[numEdges];

	for (int i= 0 ; i< numEdges; i++){
		start_edge[i] = mEdge[i].start;
		end_edge[i] = mEdge[i].end;
	}

	int *d_start_edge , *d_end_edge;
	cudaMalloc((void**)&d_start_edge , numEdges*sizeof(int));
	cudaMalloc((void**)&d_end_edge , numEdges*sizeof(int));
	cudaMemcpy(d_start_edge , start_edge, numEdges*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_end_edge , end_edge, numEdges*sizeof(int), cudaMemcpyHostToDevice);

	// Allocating and initializing the ndhdsize, nbhdvert, nbhdsign and nbhdedges on the device
	vert* mVert = g->V;

	int double_edges = 2*numEdges; 
	int* h_nbhd_size = new int[numNodes];
	int* h_nbhd_start = new int[numNodes];
 	int* h_nbhd_vert = new int[double_edges];
 	int *h_nbhd_sign = new int[double_edges];
 	int *h_nbhd_edges = new int[double_edges];

 	int local_size = 0;
 	for (int i = 0; i< numNodes ; i++){
 		h_nbhd_size[i] = mVert[i].nbhdSize;
 		h_nbhd_start[i] = 0;
 		if (i>0){
 			h_nbhd_start[i] = h_nbhd_size[i-1] + h_nbhd_start[i-1];  
 		}
 			for (int j = 0 ; j< h_nbhd_size[i] ; j++){
 				local_size = h_nbhd_start[i] + j;
 				h_nbhd_vert[local_size] = mVert[i].nbhdVert[j];
 				h_nbhd_sign[local_size] = mVert[i].sign[j];
 				h_nbhd_edges[local_size] = mVert[i].nbhdEdges[j];
 				//cout << h_nbhd_vert[local_size] << " "  << h_nbhd_sign[local_size] << " " << h_nbhd_edges[local_size] << endl;
 			}
 	}

 	int *d_nbhd_size, *d_nbhd_start, *d_nbhd_vert, *d_nbhd_sign, *d_nbhd_edges;
 	cudaMalloc((void**)&d_nbhd_size , numNodes*sizeof(int));
 	cudaMalloc((void**)&d_nbhd_start , numNodes*sizeof(int));
	cudaMalloc((void**)&d_nbhd_vert , double_edges*sizeof(int));
	cudaMalloc((void**)&d_nbhd_sign , double_edges*sizeof(int));
	cudaMalloc((void**)&d_nbhd_edges , double_edges*sizeof(int)); 	

	cudaMemcpy(d_nbhd_size , h_nbhd_size, numNodes*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbhd_start , h_nbhd_start, numNodes*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbhd_vert , h_nbhd_vert, double_edges*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbhd_sign , h_nbhd_sign, double_edges*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbhd_edges , h_nbhd_edges, double_edges*sizeof(int), cudaMemcpyHostToDevice);

	float b = g->b;
	cout << "bk file imported" << endl;

	// Allocate memory on host
	float *x = new float[numNodes];
	float *y = new float[numEdges];
	float *div_y = new float[numNodes];
	float *x_diff = new float[numNodes];
	float *grad_x_diff = new float[numEdges];
	float *tau = new float[numNodes];
	cout << "Memory allocated on host" << endl;

	// Names of all the cuda_arrays	
	read_bk<float> *d_g;
 	float *d_x, *d_y, *d_div_y, *d_x_diff, *d_grad_x_diff, *d_tau, *d_sigma;
	
	// Allocate memory on cuda	
	cudaMalloc((void**)&d_g, sizeof(read_bk<float>));
	cudaMalloc((void**)&d_x, numNodes*sizeof(float));
	cudaMalloc((void**)&d_y, numEdges*sizeof(float));
	cudaMalloc((void**)&d_div_y, numNodes*sizeof(float));
	cudaMalloc((void**)&d_x_diff, numNodes*sizeof(float));
	cudaMalloc((void**)&d_grad_x_diff, numEdges*sizeof(float));
	cudaMalloc((void**)&d_tau, numNodes*sizeof(float));
	cudaMalloc((void**)&d_sigma, numEdges*sizeof(float));
	cout << "Memory allocated on device" << endl;

	// Initialise cuda memories
	cudaMemcpy(d_g, g, sizeof(read_bk<float>), cudaMemcpyHostToDevice);	
	cudaMemset(d_x , 0, numNodes*sizeof(float));
	cudaMemset(d_y , 0, numEdges*sizeof(float));
	cudaMemset(d_div_y , 0, numNodes*sizeof(float));
	cudaMemset(d_x_diff , 0, numNodes*sizeof(float));
	cudaMemset(d_grad_x_diff , 0, numEdges*sizeof(float));
	cudaMemset(d_tau , 0, numNodes*sizeof(float));
	cudaMemset(d_sigma , 0, numEdges*sizeof(float));

	cout << "Memory initialised on device" << endl;

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
	dim3 grid = dim3(grid_x, grid_y, grid_z);

	cout << "before dt computation" << endl;
	// Pre-compute time steps
	
	d_compute_dt <<<grid, block>>> (d_tau, d_sigma, d_w, alpha, rho, d_nbhd_size, d_nbhd_edges, d_nbhd_start, numNodes, numEdges);


	cout << "time step computed" << endl;
	// Iteration
	cout << "------------------- Time loop started -------------------"  << endl;
	while (it < iter_max && gap > eps){
		// Update X
		updateX <float> <<< grid, block >>> (d_x, d_y, d_w, d_f, d_x_diff, d_div_y, d_nbhd_size, d_nbhd_start, d_nbhd_sign, d_nbhd_edges, d_tau, numNodes);

		T *sigma = new T[numNodes];
		cudaMemcpy(sigma, d_x, numNodes*sizeof(T), cudaMemcpyDeviceToHost);

		for (int i=0; i<numNodes; i++){
			cout << sigma[i] << endl;
		}

		/*  Perfectly fine upto here.. have checked d_x_diff, d_y, d_w and d_sigma.. the arrays that go into updateY  */

		// Update Y
		updateY <float> <<<grid, block >>> (d_x_diff, d_y, d_w, d_start_edge, d_end_edge, d_sigma, numEdges);

		cudaMemcpy(sigma, d_y, numNodes*sizeof(T), cudaMemcpyDeviceToHost);

		for (int i=0; i<numNodes; i++){
			cout << sigma[i] << endl;
		}

		// Update divergence of Y
		h_divergence_calculate <T> <<<grid, block>>> (d_w, d_y, d_nbhd_size, d_nbhd_start, d_nbhd_sign, d_nbhd_edges, numNodes, d_div_y);

		// Compare 0 and div_y - f
		max_vec_computation <T> <<<grid, block >>> (d_div_y, d_f, d_max_vec, numNodes);  ////  Quite sure it is right
		
		// Compute gradient of u
		h_gradient_calculate <T> <<<grid, block>>>(d_w, d_x, d_start_edge, d_end_edge, numEdges, d_grad_x);
		
		// Compute L1 norm of gradient of u
		cublasSasum(handle, numNodes, d_grad_x, 1, &x_norm);  /// seems to add up the value

		// Compute scalar product
		cublasSdot(handle, numNodes, d_x, 1, d_f, 1, &xf);	/// seems to do to the dot product
		 
		// Summing up the max_vec
		cublasSasum(handle, numNodes, d_max_vec, 1, &max_val); // works just fine... no problem here
		
		// Compute gap
		gap = (xf + x_norm + max_val) / numEdges;
		cout << "Iteration = " << it << endl << endl;
		cout << "Gap = " << gap << endl << endl;
		it = it + 1;
	}
	
	/*T *sigma = new T[numNodes];
	cudaMemcpy(sigma, d_x, numNodes*sizeof(T), cudaMemcpyDeviceToHost);

	for (int i=0; i<numNodes; i++){
		cout << sigma[i] << endl;
	}*/

	
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
	delete[] start_edge;
	delete[] end_edge;
	delete[] h_nbhd_size;
	delete[] h_nbhd_vert;
	delete[] h_nbhd_sign;
	delete[] h_nbhd_edges;
	
	cudaFree(d_grad_x);
	cudaFree(d_max_vec);
	cudaFree(d_gap_vec);


    return 0;
}
