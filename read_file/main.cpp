#include <iostream>
#include "read_bk.h"
using namespace std;

int main(int argc, char **argv)
{
    if (argc <= 1)
	{
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
    }
    
    read_bk<float> *g = new read_bk<float>(argv[1]); 
    int numNodes  = g->nNodes;
    int numEdges = g->nEdges;
    float *f = g->f;
    float **w = g->w;
    int **ord = g->ord;
    cout << "Number of edges is " << numEdges <<endl;

    // For testing only:
    /*cout << "\nIn main\n" <<endl;
    for(int i = 0; i<numEdges; i++)
    {
        cout<<"ord is "<< ord[i][0] <<", "<< ord[i][1] << endl;
    } 
    */
    /*
    
    cout << "Printing f" <<endl;
    for (int n = 0; n<numNodes; n++)
        cout<<"f_"<<n<<" = " << f[n]<< endl;
    
    cout<< "Printing w" <<endl;
    for (int i = 0; i<numNodes; i++)
        for (int j = 0; j<numNodes; j++)
            cout<<"w_{"<<i<<","<<j<<"} = " << w[i][j] << endl;
    */
    delete g;
    
    return 0;
}
