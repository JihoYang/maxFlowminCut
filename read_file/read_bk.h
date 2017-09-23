#include <iostream>
#include <fstream>
#include <vector>

#ifndef _READ_BK_H
#define _READ_BK_H


// Data structures to store vertices and nodes
// For Divergence
struct vert
{
    std::vector<int> nbhdVert;
    // sign contains 1 or -1
	std::vector<int> sign;
	std::vector<int> nbhdEdges;
};

// Consider undirected graph!
// For Gradient
template <class P>
struct edge
{
    // start is supposed to be smaller than end
    int start = 0, end = 0;
    P* weight;
};



template <class T>
class read_bk
{
    public:    
        // Variables
        int nNodes = 0;
        int nEdges = 0;
        // Vector of weights per node
        T *f;
        // Array of weights, size = nNodes * nNodes
        T **w;
        // Array containing nodes that form an edge
        // Given an edge "i", ord[i][0] is the smaller node 
        // that composes it and ord[i][1] the bigger one.     
        int **ord;
        
        // Methods
        // Allocate memory for dynamic array
        void init_graph(int numberNodes, int numberEdges);
        void free_memory();
        bool readFile(char* fileName);

        read_bk(char* filename);
        
        ~read_bk();
        
};
#endif
