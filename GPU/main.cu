//////////////////////////////////////////////////////////////////////////
//																		//
//		GPU accelerated max flow min cut graph problem solver			//
//																		//
//		Written by: Apoorva Gupta										//
//					Jorge Salazar										//
//					Jiho Yang											//
//																		//
//		Final update: 30/09/2017										//
//																		//
//////////////////////////////////////////////////////////////////////////

// TODO: Something wrong with the pointer!

#include <iostream>
#include <vector>
#include <time.h>
#include "read_bk.h"
#include "primal_dual.cuh"
#include "mathOperations.cuh"
#include "postProcessing.cuh"
#include "helper.cuh"
#include <string.h>	
#include <cublas_v2.h>

//# define T float
//# define FLOAT

#define T double
#define DOUBLE


using namespace std;

template<class S>
void printDevice(S* d_arr, int num_elem, char* s)
{
	S* temp = new S[num_elem] ;
	cudaMemcpy(temp, d_arr, num_elem*sizeof(S), cudaMemcpyDeviceToHost);
	for(int i = 0; i<num_elem; i++)
	{
		cout<< s << "_"<<i<<" is "<< temp[i] <<endl; 
	} 
	delete temp; 
}

int main(int argc, char **argv)
{
    if (argc <= 1)
	{
		printf("Usage: %s <filename> -alpha <value> - rho <value> -it <maximum number of iterations>\n", argv[0]);
		return 1;
    }
	// Start time
	clock_t tStart = clock();
	// Parameters
	T alpha = 1;
	T rho = 1;
	T gap = 1;
	T eps = 1E-4;
	int it  = 0;
	int iter_max = 100000;
	T xf;
	T x_norm;
	T max_flow;
	T max_val;
	//const char *method = "PD_CPU";
	// Command line parameters
	getParam("alpha", alpha, argc, argv);
	cout << "alpha: " << alpha << endl;
	getParam("rho", rho, argc, argv);
	cout << "rho: " << rho << endl;
	getParam("it", iter_max, argc, argv);
	cout << "it: " << iter_max << endl;
	// Import bk file    
	read_bk<T> *g = new read_bk<T>(argv[1]); 	
	int numNodes  = g->nNodes;
	int numEdges = g->nEdges;
	T *f = g->f;
	T *w = g->w;
	//vert* mVert = g->V;
	T b = g->b;

	cout << "bk file imported in HOST"  << endl;
	
	// Allocating and initializing f and w on the device
	T *d_f , *d_w;
	cudaMalloc((void**)&d_f , numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMalloc((void**)&d_w , numEdges*sizeof(T));													//CUDA_CHECK;
	cudaMemcpy(d_f , f, numNodes*sizeof(T), cudaMemcpyHostToDevice);								//CUDA_CHECK;
	cudaMemcpy(d_w , w, numEdges*sizeof(T), cudaMemcpyHostToDevice);								//CUDA_CHECK;

	cout << "Allocation and Initialization of f and w on DEVICE completed" << endl << endl;

	int* start_edge = g->edge_start;
	int* end_edge = g->edge_end;
	int *d_start_edge , *d_end_edge;
	cudaMalloc((void**)&d_start_edge , numEdges*sizeof(int));										//CUDA_CHECK;
	cudaMalloc((void**)&d_end_edge , numEdges*sizeof(int));											//CUDA_CHECK;
	cudaMemcpy(d_start_edge , start_edge, numEdges*sizeof(int), cudaMemcpyHostToDevice);			//CUDA_CHECK;
	cudaMemcpy(d_end_edge , end_edge, numEdges*sizeof(int), cudaMemcpyHostToDevice);				//CUDA_CHECK;

	cout << "Allocation and Initialization of start_edge and end_edge on DEVICE completed" << endl << endl;

	int double_edges = 2 * numEdges;
	int* h_nbhd_size = &(g->h_nbhd_size[0]); 
	int* h_nbhd_start = &(g->h_nbhd_start[0]) ;
 	int* h_nbhd_vert = &(g->h_nbhd_vert[0]);
 	int *h_nbhd_sign = &(g->h_nbhd_sign[0]);
	int *h_nbhd_edges = &(g->h_nbhd_edges[0]);
	
 	int *d_nbhd_size, *d_nbhd_start, *d_nbhd_vert, *d_nbhd_sign, *d_nbhd_edges;
 	cudaMalloc((void**)&d_nbhd_size , numNodes*sizeof(int));										//CUDA_CHECK;
 	cudaMalloc((void**)&d_nbhd_start , numNodes*sizeof(int));										//CUDA_CHECK;
	cudaMalloc((void**)&d_nbhd_vert , double_edges*sizeof(int));									//CUDA_CHECK;
	cudaMalloc((void**)&d_nbhd_sign , double_edges*sizeof(int));									//CUDA_CHECK;
	cudaMalloc((void**)&d_nbhd_edges , double_edges*sizeof(int)); 									//CUDA_CHECK;

	cudaMemcpy(d_nbhd_size , h_nbhd_size, numNodes*sizeof(int), cudaMemcpyHostToDevice);			//CUDA_CHECK;
	cudaMemcpy(d_nbhd_start , h_nbhd_start, numNodes*sizeof(int), cudaMemcpyHostToDevice);			//CUDA_CHECK;
	cudaMemcpy(d_nbhd_vert , h_nbhd_vert, double_edges*sizeof(int), cudaMemcpyHostToDevice);		//CUDA_CHECK;
	cudaMemcpy(d_nbhd_sign , h_nbhd_sign, double_edges*sizeof(int), cudaMemcpyHostToDevice);		//CUDA_CHECK;
	cudaMemcpy(d_nbhd_edges , h_nbhd_edges, double_edges*sizeof(int), cudaMemcpyHostToDevice);		//CUDA_CHECK;

	cout << "Allocation and Initialization of  d_nbhd_size, d_nbhd_start, d_nbhd_vert, d_nbhd_sign, d_nbhd_edges and end_edge on DEVICE completed" << endl << endl;

	delete[] h_nbhd_size;
	delete[] h_nbhd_vert;
	delete[] h_nbhd_sign;
	delete[] h_nbhd_edges;

	// Names of all the cuda_arrays	
 	T *d_x, *d_y, *d_div_y, *d_x_diff, *d_grad_x_diff, *d_tau, *d_sigma;
 	T *d_grad_x, *d_max_vec, *d_gap_vec;
	
	// Allocate memory on cuda	
	cudaMalloc((void**)&d_x, numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMalloc((void**)&d_y, numEdges*sizeof(T));													//CUDA_CHECK;
	cudaMalloc((void**)&d_div_y, numNodes*sizeof(T));												//CUDA_CHECK;
	cudaMalloc((void**)&d_x_diff, numNodes*sizeof(T));												//CUDA_CHECK;
	cudaMalloc((void**)&d_grad_x_diff, numEdges*sizeof(T));											//CUDA_CHECK;
	cudaMalloc((void**)&d_tau, numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMalloc((void**)&d_sigma, numEdges*sizeof(T));												//CUDA_CHECK;
	cudaMalloc((void**)&d_grad_x, numEdges*sizeof(T));												//CUDA_CHECK;
	cudaMalloc((void**)&d_max_vec, numNodes*sizeof(T));												//CUDA_CHECK;
	cudaMalloc((void**)&d_gap_vec, numNodes*sizeof(T));												//CUDA_CHECK;
	// Initialise cuda memories
	cudaMemset(d_x , 0, numNodes*sizeof(T));														//CUDA_CHECK;
	cudaMemset(d_y , 0, numEdges*sizeof(T));														//CUDA_CHECK;
	cudaMemset(d_div_y , 0, numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMemset(d_x_diff , 0, numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMemset(d_grad_x_diff , 0, numEdges*sizeof(T));												//CUDA_CHECK;
	cudaMemset(d_tau , 0, numNodes*sizeof(T));														//CUDA_CHECK;
	cudaMemset(d_sigma , 0, numEdges*sizeof(T));													//CUDA_CHECK;
	cudaMemset(d_grad_x, 0 , numEdges*sizeof(T));													//CUDA_CHECK;
	cudaMemset(d_max_vec, 0 , numNodes*sizeof(T));													//CUDA_CHECK;
	cudaMemset(d_gap_vec, 0 , numNodes*sizeof(T));													//CUDA_CHECK;

	cout << "Memory Allocated and initiaized for temperory arrays on DEVICE" << endl << endl;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cout << "handle for BLAS operations created" << endl << endl;

	dim3 block = dim3(1024,1,1);
	int grid_x = ((max(numNodes, numEdges) + block.x - 1)/block.x);
	int grid_y = 1;
	int grid_z = 1;
	dim3 grid = dim3(grid_x, grid_y, grid_z);

	cout << "grid and block dimensions calculated" << endl << endl;

	d_compute_dt <<<grid, block>>> (d_tau, d_sigma, d_w, alpha, rho, d_nbhd_size, d_nbhd_edges, d_nbhd_start, numNodes, numEdges); 									//CUDA_CHECK;

	cout << "tau and sigma calculation completed on the DEVICE" << endl << endl;

	// Iteration
	cout << "------------------- Time loop started -------------------"  << endl;
	while (it < iter_max && gap > eps){
		updateX <T> <<< grid, block >>> (d_x, d_y, d_w, d_f, d_x_diff, d_div_y, d_nbhd_size, d_nbhd_start, d_nbhd_sign, d_nbhd_edges, d_tau, numNodes);				//CUDA_CHECK;

		// Update Y
		updateY <T> <<<grid, block >>> (d_x_diff, d_y, d_w, d_start_edge, d_end_edge, d_sigma, numEdges);															//CUDA_CHECK;

		if (it % 1000 == 0){

			// Update divergence of Y	
			h_divergence_calculate <T> <<<grid, block>>> (d_w, d_y, d_nbhd_size, d_nbhd_start, d_nbhd_sign, d_nbhd_edges, numNodes, d_div_y);							//CUDA_CHECK;

			// Compare 0 and div_y - f
			max_vec_computation <T> <<<grid, block >>> (d_div_y, d_f, d_max_vec, numNodes);  																			//CUDA_CHECK;
			
			// Compute gradient of u
			h_gradient_calculate <T> <<<grid, block>>>(d_w, d_x, d_start_edge, d_end_edge, numEdges, d_grad_x);															//CUDA_CHECK;

			#ifdef FLOAT
				// Compute L1 norm of gradient of u
				cublasSasum(handle, numEdges, d_grad_x, 1, &x_norm);  								//CUDA_CHECK;

				// Compute scalar product 
				cublasSdot(handle, numNodes, d_x, 1, d_f, 1, &xf);	                                //CUDA_CHECK;
				 
				// Summing up the max_vec
				cublasSasum(handle, numNodes, d_max_vec, 1, &max_val); 								//CUDA_CHECK;
			
			#else
				cublasDasum(handle, numEdges, d_grad_x, 1, &x_norm);								//CUDA_CHECK;
				
				cublasDdot(handle, numNodes, d_x, 1, d_f, 1, &xf);									//CUDA_CHECK;
				
				cublasDasum(handle, numNodes, d_max_vec, 1, &max_val);								//CUDA_CHECK;
			
			#endif
			
			// Compute gap
			gap = (xf + x_norm + max_val) / (T)numEdges;
			cout << "Iteration = " << it << endl << endl;
			cout << "Gap = " << gap  << "  xf  " << xf << "  x_norm  " << x_norm << "  max_val  " << max_val << "  max_flow  " << (xf + x_norm + b) << endl;
		}
		
		it = it + 1;
	}

	// Round solution
	//round_solution <T> <<<grid, block>>> (d_x, numNodes);											//CUDA_CHECK;
	
	// End time
	clock_t tEnd = clock();
	// Compute max flow
	//h_gradient_calculate <T> <<<grid, block>>>(d_w, d_x, d_start_edge, d_end_edge, numEdges, d_grad_x);		//CUDA_CHECK;

	#ifdef FLOAT
		cublasSasum(handle, numEdges, d_grad_x, 1, &x_norm);  										//CUDA_CHECK;

		cublasSdot(handle, numNodes, d_x, 1, d_f, 1, &xf);	                           				//CUDA_CHECK;
		
	#else
		cublasDasum(handle, numEdges, d_grad_x, 1, &x_norm);										//CUDA_CHECK;
		
		cublasDdot(handle, numNodes, d_x, 1, d_f, 1, &xf);											//CUDA_CHECK;
	#endif


	//printDevice<float>(d_x, numNodes, "x_");
	cout  << "  xf  " << xf << "  x_norm  " << x_norm << "  b  " << b<< endl;

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
	
	cudaFree(d_f);																					//CUDA_CHECK;
	cudaFree(d_w);																					//CUDA_CHECK;
	cudaFree(d_start_edge);																			//CUDA_CHECK;
	cudaFree(d_end_edge);																			//CUDA_CHECK;
	cudaFree(d_nbhd_size);																			//CUDA_CHECK;
	cudaFree(d_nbhd_start);																			//CUDA_CHECK;
	cudaFree(d_nbhd_vert);																			//CUDA_CHECK;
	cudaFree(d_nbhd_sign);																			//CUDA_CHECK;
	cudaFree(d_nbhd_edges);																			//CUDA_CHECK;
	cudaFree(d_x);																					//CUDA_CHECK;
	cudaFree(d_y);																					//CUDA_CHECK;
	cudaFree(d_div_y);																				//CUDA_CHECK;
	cudaFree(d_x_diff);																				//CUDA_CHECK;
	cudaFree(d_grad_x_diff);																		//CUDA_CHECK;
	cudaFree(d_tau);																				//CUDA_CHECK;
	cudaFree(d_sigma);																				//CUDA_CHECK;
	cudaFree(d_grad_x);																				//CUDA_CHECK;
	cudaFree(d_max_vec);																			//CUDA_CHECK;
	cudaFree(d_gap_vec);																			//CUDA_CHECK;

    return 0;
}
