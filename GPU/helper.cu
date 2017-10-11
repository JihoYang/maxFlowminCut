#include "helper.cuh"
#include <cstdlib>
#include <iostream>
using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using namespace std;

// parameter processing: template specialization for T=bool
template<class T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

// cuda error checking
string prev_file = "";
int prev_line = 0;
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

int output_data(char** argv, int* data_array, int length, std::string array_name)
{
    const char* file_name = array_name.c_str();
	ofstream myfile;
	myfile.open (file_name , ios::in | ios::trunc);
	myfile << argv[1] << endl;
	for (int i =0 ; i<length; i++){	
		myfile << data_array[i] << endl;
	}
	myfile.close();
	return 0;
}

int output_data(char** argv, int* data_array, int length, std::string array_name);
template bool getParam(std::string param, int &var, int argc, char **argv);
template bool getParam(std::string param, float &var, int argc, char **argv);
template bool getParam(std::string param, double &var, int argc, char **argv);
