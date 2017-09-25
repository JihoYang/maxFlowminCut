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
    ~vert()
    {
        std::vector<int>().swap(nbhdVert);
        std::vector<int>().swap(sign);
        std::vector<int>().swap(nbhdEdges);
    } 
};

// Consider undirected graph!
// For Gradient
struct edge
{
    // start is supposed to be smaller than end
    int start= 0, end= 0;
};


template <class T>
class read_bk
{
    public:    
        // Variables
        int nNodes = 0;
        int nEdges = 0;
        
        // Structures for vertices and edges
        vert *V;
        edge *E;
        // Vector of weights per node, size = nNodes
        T *f;
        // Array of weights, size = nEdges
        T *w;
        
        // Methods
        // Allocate memory for dynamic array
        void init_graph(int numberNodes, int numberEdges);
        void free_memory();
        bool readFile(char* fileName);

        read_bk(char* filename);
        
        ~read_bk();
        
};
#endif