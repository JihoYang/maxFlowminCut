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
	
// TODO: the code diverges - check for the correctness of primal-dual algorithm and grad/div computation

#include <iostream>
#include <vector>
#include <time.h>
#include "read_bk.h"
#include "primal_dual.h"
#include "mathOperations.h"
#include "postProcessing.h"
#include <string.h>

using namespace std;

template <class S>
void printResults(S* results, int num_elem, char* name)
{
	for(int i = 0; i< num_elem; i++)
		cout<<name<<"_"<<i<<" is "<<results[i]<<endl;
}

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
	int	  it  = 0;
	int iter_max = 1000;
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
	// Allocate memory
	float *x = new float[numNodes];
	float *y = new float[numEdges];
	float *div_y = new float[numNodes];
	float *x_diff = new float[numNodes];
	float *grad_x_diff = new float[numEdges];
	float *tau = new float[numNodes];
	float *sigma = new float[numEdges];
	float xf;
	float x_norm;
	float max_flow;
	// Initialise x and y
	memset(x, 0, sizeof(float)*numNodes);
	memset(y, 0, sizeof(float)*numEdges);
	// Pre-compute time steps
	compute_dt <float> (tau, sigma, w, alpha, rho, mVert, numNodes, numEdges);
	//printResults(tau, numNodes, "tau");
	//printResults(sigma, numEdges, "sigma");
	//cout << "index="<<e <<", print grad "<<grad[e]<<endl;
	// Iteration
	cout << "------------------- Time loop started -------------------"  << endl;
	while (it < iter_max && gap > eps){
		// Update X
		updateX <float> (w, mVert, x, tau, div_y, y, f, x_diff, numNodes);
	
		//for (int i = 0 ; i<numNodes ; i++){
		//	cout << "x:" << i << "  " << x[i] << endl;
		//}
		
		// Update Y for next iteration
		updateY <float> (w, x, mEdge, y, sigma, x_diff, grad_x_diff, numEdges);
		//printResults(x, numNodes, "x");
		//printResults(y, numEdges, "y");
		///printResults(x_diff, numNodes, "x_diff");
		//printResults(div_y, numNodes, "div_y");
		//for (int i = 0 ; i<numEdges ; i++){
		//	cout << "y:" << i << "  " << y[i] << endl;
		//}

		// Compute gap
		compute_gap <float> (w, mEdge, mVert, x, y, f, gap, x_norm, xf, numNodes, numEdges);
		cout << "Iteration = " << it << endl << endl;
		cout << "Gap = " << gap << endl << endl;
		it = it + 1;
		
	}
	// End time
	clock_t tEnd = clock();
	// Compute max flow
	float* grad_x = new float[numNodes]; 
	roundVector<float>(x, numNodes);
	printResults<float>(x, numNodes, "x_final");
	compute_scalar_product<float>(x, f, xf, numNodes);
	gradient_calculate<float>(w, x, mEdge, numEdges, grad_x);
	// Compute scalar product
	// Compute L1 norm of gradient of u
	compute_L1<float>(grad_x, x_norm, numEdges);
	//cout << " Xf = " << xf << " x_norm = " << x_norm << " max_val = " << max_val << endl;
	// Compute gap

	max_flow = xf + x_norm + b;
	cout << "xf="<<xf<<" x_norm="<<x_norm<<" b="<<b<<endl;

	cout << "Max flow = " << max_flow << endl << endl;


	// Program exit messages
	if (it == iter_max) cout << "ERROR: Maximum number of iterations reached" << endl << endl;
	cout << "------------------- End of program -------------------"  << endl << endl;
	cout << "Execution Time = " << (double)1000*(tEnd - tStart)/CLOCKS_PER_SEC << " ms" << endl << endl;
	//Export results
	export_result <float> (method, x, numNodes);
	// Free memory    
    delete g;
	delete []x;
	delete []y;
	delete []x_diff;
	delete []div_y;
	delete []grad_x_diff;
	delete []tau;
	delete []sigma;	
	delete [] grad_x;
	delete [] f;
	delete [] w; 

    return 0;
}
