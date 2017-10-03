#include <iostream>
#include <fstream>
#include <vector>

#ifndef _READ_BK_H
#define _READ_BK_H


// Data structures to store vertices and nodes
// For Divergence
struct vert
{
    public:
        std::vector<int> _nbhdVert;
        // sign contains 1 or -1
        std::vector<int> _sign;
        std::vector<int> _nbhdEdges;
        int* nbhdVert;
        int* sign;
        int* nbhdEdges;
        int nbhdSize;
        ~vert()
        {
            std::vector<int>().swap(_nbhdVert);
            std::vector<int>().swap(_sign);
            std::vector<int>().swap(_nbhdEdges);
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
        int *edge_start;
        int *edge_end; 
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
    private:
        void assign_pointers();        
};
#endif

