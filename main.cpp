//////////////////////////////////////////////////////////////////////////
//																		//
//		GPU accelerated max flow min cut graph problem solver			//
//																		//
//		Written by: Apoorva Gupta										//
//					Jorge Salazar										//
//					Jiho Yang											//
//																		//
//		Final update: 25/09/2017										//
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
	// Initialise x and y
	memset(x, 0, sizeof(float)*numNodes);
	memset(y, 0, sizeof(float)*numEdges);
	// Pre-compute time steps
	compute_dt <float> (tau, sigma, w, alpha, rho, mVert, numNodes, numEdges);
	// Iteration
	cout << "------------------- Time loop started -------------------"  << endl;
	while (it < iter_max && gap > eps){
		// Update X
		updateX <float> (w, mVert, x, tau, div_y, y, f, x_diff, numNodes);
		// Compute gap
		compute_gap <float> (w, mEdge, x, f, div_y, gap, numNodes, numEdges);
		cout << "Iteration = " << it << endl << endl;
		cout << "Gap = " << gap << endl << endl;
		it = it + 1;
		// Update Y for next iteration
		updateY <float> (w, x, mEdge, y, sigma, x_diff, grad_x_diff, numEdges);
	}
	// End time
	clock_t tEnd = clock();
	// Program exit messages
	if (it == iter_max) cout << "ERROR: Maximum number of iterations reached" << endl << endl;
	cout << "------------------- End of program -------------------"  << endl << endl;
	cout << "Execution Time = " << (double)1000*(tEnd - tStart)/CLOCKS_PER_SEC << " ms" << endl << endl;
	//Export results
	const char* dt_tau = "dt_tau";
	const char* dt_sigma = "dt_sigma";
	export_result <float> (method, x, numNodes);
	export_result <float> (dt_tau, tau, numNodes);
	export_result <float> (dt_sigma, sigma, numEdges);
	// Free memory    
    delete g;
	delete []x;
	delete []y;
	delete []x_diff;
	delete []div_y;
	delete []grad_x_diff;
	delete []tau;
	delete []sigma;	

    return 0;
}
