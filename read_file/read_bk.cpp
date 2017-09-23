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
    // Set memory for f(initial node setup), 
	// w(weights on the edges) and ord(composition array of edges)
	nNodes = numberNodes;
	nEdges = numberEdges;
    f = new T[nNodes];
	w = new T*[nNodes];
	ord = new int*[nEdges]; 
	// Set values to 0
	for(int i = 0; i<nEdges; i++)
	{
		ord[i] = new int[2];
		ord[i][0] = 0; 
		ord[i][1] = 0; 
	}	
	for(int i = 0; i<nNodes; i++)
	{	
		f[i] = 0;
		w[i] = new T[nNodes]; 			  
		for(int j = 0; j<nNodes; j++)
		{
			w[i][j] = 0;
		}	
	}	
}

template<class T>
void read_bk<T>::free_memory()
{
	delete[] f;
	for(int i = 0; i<nNodes; i++)
		delete[] w[i];
	for(int i = 0; i<nEdges; i++)
		delete[] ord[i];
	delete[] w; 
	delete[] ord; 
}


template<class T>
bool read_bk<T>::readFile(char *filename)
{
	const int MAX_LINE_LEN = 100;
	char line[MAX_LINE_LEN];
	int min= 0, max= 0;
	int declaredNumOfNodes, declaredNumOfEdges, nodeId1, nodeId2;
	int currentNumOfEdges = 0;
	char c;
	// Maybe change this to only integers, depending on the problem..
	double capacity, capacity2, a, b;
	int numLines=0;
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
				sscanf(line, "%c %d %d", &c, &declaredNumOfNodes, &declaredNumOfEdges);
				cout << "Number of nodes is "<< declaredNumOfNodes << " and number of edges is "<< declaredNumOfEdges << endl;
				init_graph(declaredNumOfNodes, declaredNumOfEdges);
				break;

			//
			// Read Nodes
			//
			case 'n':
				sscanf(line, "%c %d %lf %lf ", &c, &nodeId1, &capacity, &capacity2);
				if (capacity == 0.f && capacity2 == 0.f) 
				{
					cout<< "In node "<<nodeId1<<" the excess is "<<capacity<<" and the deficit is "<< capacity2 << endl; 
					break;
				}
				// Add values to f per node(if connected to source or sink)		
				else f[nodeId1] += capacity2 - capacity;
				cout<< "In node "<<nodeId1<<" the excess is "<<capacity<<" and the deficit is "<< capacity2 << endl; 
				break;

			//
			// Read Edges
			//
			case 'a':
				if (currentNumOfEdges >= declaredNumOfEdges) 
				{
					fprintf(stdout, "Number of edges in file does not match (Line %d)\n", numLines);
					return false;
				}
				capacity = 0.f; capacity2 = 0.f;
				sscanf(line, "%c %d %d %lf %lf", &c, &nodeId1, &nodeId2, &capacity, &capacity2);
				cout<< "Edge with node1 "<<nodeId1<<" and node2 "<<nodeId2<<" capacity n1n2:  "<< capacity 
							<<", and capacity n2n1: "<<capacity2<< endl; 

				// Store edges ordering
				if(nodeId1 < nodeId2){min = nodeId1; max = nodeId2;}
				else{min = nodeId2; max = nodeId1;} 
				ord[currentNumOfEdges][0] = min;
				ord[currentNumOfEdges][1] = max;    	

				// Add values to f and w per edge
				a = capacity/2.f; b = capacity2/2.f;
				f[nodeId1] += b - a;
				f[nodeId2] += a - b;
				w[nodeId1][nodeId2] = a + b;
				w[nodeId2][nodeId1] = a + b;

				currentNumOfEdges++;
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

