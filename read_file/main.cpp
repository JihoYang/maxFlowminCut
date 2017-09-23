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
    float *w = g->w;
    vert* mVert = g->V;
    edge<float>* mEdge = g->E;

    cout << "\n-- main --\n "<<endl;
    int local_size;
    for(int i = 0; i<numNodes; i++)
    {
        local_size = mVert[i].nbhdVert.size();
        cout << "Local size is "<< local_size<< endl;
        cout<<"Vertex "<< i <<" has "<< local_size  <<" nbhrs" << endl;
        for(int j = 0 ; j < local_size; j++)
        {
            cout<<"Vertex "<< i <<" has nbhd edge "<< mVert[i].nbhdEdges[j] <<endl;
        } 
    } 
    
    for (int n = 0; n<numEdges; n++)
        cout<<"Edge has start "<< mEdge[n].start  <<" and end is " << mEdge[n].end << endl;
    
    delete g;
    
    return 0;
}
