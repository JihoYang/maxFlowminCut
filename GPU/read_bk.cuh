#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/device_vector.h>

#ifndef _READ_BK_H
#define _READ_BK_H


// Data structures to store vertices and nodes
// For Divergence
struct vert
{
    thrust::device_vector<int> nbhdVert;
    // sign contains 1 or -1
	thrust::device_vector<int> sign;
    thrust::device_vector<int> nbhdEdges;
    ~vert()
    {
        thrust::device_vector<int>().swap(nbhdVert);
        thrust::device_vector<int>().swap(sign);
        thrust::device_vector<int>().swap(nbhdEdges);
    } 
};

// Consider undirected graph!
// For Gradient
struct edge
{
    // start is supposed to be smaller than end
    int start, end;
};


template <class T>
class read_bk
{
    public:    
        // Variables
        int nNodes;
        int nEdges;
        
        // Structures for vertices and edges
        vert *V;
        edge *E;
        // Vector of weights per node, size = nNodes
        T *f;
        // Array of weights, size = nEdges
        T *w;
		// Constant factor for energy computation
		T b;
        
        // Methods
        // Allocate memory for dynamic array
        void init_graph(int numberNodes, int numberEdges);
        void free_memory();
        bool readFile(char* fileName);

        read_bk(char* filename);
        
        ~read_bk();
        
};
#endif
