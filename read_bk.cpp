#include "read_bk.h"
using namespace std;

template<class T>
read_bk<T>::read_bk(char *filename)
{
	readFile(filename);
} 

template<class T>
read_bk<T>::~read_bk()
{
	free_memory();
}

template<class T>
void read_bk<T>::init_graph(int numberNodes, int numberEdges)
{
	nNodes = numberNodes;
	nEdges = numberEdges;
	// Allocate memory for vertex and edge structures
	V = new vert[nNodes];
	E = new edge[nEdges];
    // Allocate memory for f(initial node setup), 
	// w(weights on the edges)
	f = new T[nNodes];
	w = new T[nEdges];
	
	for(int i = 0; i<nNodes; i++) f[i] = 0;	
	for(int i = 0; i<nEdges; i++) w[i] = 0;	
}

template<class T>
void read_bk<T>::free_memory()
{
	delete[] V;
	delete[] E;   
	delete[] f;
	delete[] w;
}


template<class T>
bool read_bk<T>::readFile(char *filename)
{
	int min= 0, max= 0;
	int numNodes, numEdges, nodeId1, nodeId2;
	int currNumEdges = 0;
	int numLines=0;
	int sign = 1;
	const int MAX_LINE_LEN = 100;
	char line[MAX_LINE_LEN];
	char c;
	// Maybe change this to only integers, depending on the problem..
	double capacity, capacity2, a, b;
	FILE *pFile;
	
	if ((pFile = fopen(filename, "r")) == NULL) 
	{
		fprintf(stdout, "Could not open file %s\n", filename);
		return false;
	}

	while (fgets(line, MAX_LINE_LEN, pFile) != NULL)
	{
		numLines++;
		switch (line[0])
	    {
			case 'c':   // Comments
			case '\n':  // Line jump
			case '\0':  // Null character/ end of line
			default:
				break;
				
            //
            //  Read size of nodes and edges
            //
			case 'p':
				sscanf(line, "%c %d %d", &c, &numNodes, &numEdges);
				//cout << "Number of nodes is "<< numNodes << " and number of edges is "<< numEdges << endl;
				init_graph(numNodes, numEdges);
				break;

			//
			// Read Nodes
			//
			case 'n':
				sscanf(line, "%c %d %lf %lf ", &c, &nodeId1, &capacity, &capacity2);
				if (capacity == 0.f && capacity2 == 0.f) 
				{
					//cout<< "In node "<<nodeId1<<" the excess is "<<capacity<<" and the deficit is "<< capacity2 << endl; 
					break;
				}
				// Add values to f per node(if connected to source or sink)		
				else f[nodeId1] += capacity2 - capacity;
				//cout<< "In node "<<nodeId1<<" the excess is "<<capacity<<" and the deficit is "<< capacity2 << endl; 
				break;

			//
			// Read Edges
			//
			case 'a':
				if (currNumEdges >= numEdges) 
				{
					fprintf(stdout, "Number of edges in file does not match (Line %d)\n", numLines);
					return false;
				}
				capacity = 0.f; capacity2 = 0.f;
				sscanf(line, "%c %d %d %lf %lf", &c, &nodeId1, &nodeId2, &capacity, &capacity2);
				//cout<< "Edge with node1 "<<nodeId1<<" and node2 "<<nodeId2<<" capacity n1n2:  "<< capacity <<", and capacity n2n1: "<<capacity2<< endl; 

				// Store edges ordering
				if(nodeId1 < nodeId2)
				{
					min = nodeId1; 
					max = nodeId2;
					sign = 1;
				}
				else
				{
					min = nodeId2; 
					max = nodeId1;
					sign = -1;
				}
				// Info for edge 
				E[currNumEdges].start = min;
				E[currNumEdges].end = max; 	
				// Info for node1			
				V[nodeId1].nbhdVert.push_back(nodeId2);
				V[nodeId1].sign.push_back(sign);
				V[nodeId1].nbhdEdges.push_back(currNumEdges);
				// for node2
				V[nodeId2].nbhdVert.push_back(nodeId1);
				V[nodeId2].sign.push_back(-sign);
				V[nodeId2].nbhdEdges.push_back(currNumEdges);

				// Add values to f and w per edge
				a = capacity/2.f; b = capacity2/2.f;
				f[nodeId1] += b - a;
				f[nodeId2] += a - b;
				w[currNumEdges]  = a + b;

				currNumEdges++;
				break;
		}
	}

	
	fclose(pFile);
	pFile = NULL;
	
	return true;
}

// Tell the compiler which T we are going to use
template class read_bk<float>;
template class read_bk<double>;

