#ifndef HELPER_CUH
#define HELPER_CUH

#include <cuda_runtime.h>
#include <ctime>
#include <string>
#include <sstream>

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc)) continue;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);



#endif  // AUX_H
