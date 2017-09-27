CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
FIND_PACKAGE(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(main main.cu mathOperations.cpp postProcessing.cpp primal_dual.cpp read_bk.cpp)