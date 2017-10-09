#ifndef HELPER_CUH
#define HELPER_CUH

#include <cuda_runtime.h>
#include <ctime>
#include <string>
#include <sstream>

// parameter processing
template<class T>
bool getParam(std::string param, T &var, int argc, char **argv);

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#endif  // AUX_H
