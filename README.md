# maxFlowminCut

### GPU accelerated first order primal-dual algorithm for solving convex optimization problems, and its application in maximum flow min cut graph problem

This work was conducted as part of "Practical Course: GPU Programming in Computer Vision" at Technische Universität München (summer semester 2017).

Three MSc Computational Science & Engineering students contributed to this work (in alphabetical order):

* Apoorv Gupta
* Jorge Salazar 
* Jiho Yang

Four directories are present in this repository: CPU, GPU, IBFS, and graphs

* CPU : Contains sequential version of the code
* GPU : Contains CUDA implemented GPU-parallelised version of the code
* IBFS : Contains IBFS (Incremental Breadth First Search Algorithm) code, which was served as a benchmark solver (http://www.cs.tau.ac.il/~sagihed/ibfs/index.html)
* graphs : Contains some of test graphs in .bk file format. These test cases were taken from: http://www.cs.tau.ac.il/~sagihed/ibfs/benchmark.html

#### Requirements

* UNIX installed machine with NVIDIA GPU
* NVIDIA CUDA Compiler (nvcc)
* python 3 with matplotlib 

#### Compilation

* CPU : Use Makefile  
```
make
```
        
* GPU : Cmake constructed 
```
cd cmake_build
rm CMakeCache.txt
cmake ..
make
```

#### Binary execution 

* CPU: 
```
./sim <test.bk>
```
For instance 
```
./sim ../graphs/test.bk
```

* GPU: 
```
./main <test.bk> -alpha <value> -rho <value> -it <value>
```

where "alpha" and "rho" are hyperparameters for computing time steps, and "it" is the maximum number of iterations.
Default values for alpha, rho, and it are 1, 1, and 10000, respectively.
       
For instance 
```
./main ../../graphs/test.bk -alpha 1 -rho 1 -it 1000
```

#### Automation with bash script

The compilation, execution, and some of the visualisations are automated via bash script.

* Usage: 
```
./script
```

