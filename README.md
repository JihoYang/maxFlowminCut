# maxFlowminCut

### GPU accelerated first order primal-dual algorithm for solving convex optimization problems, and its application in maximum flow min cut graph problem

This work was conducted as part of "Practical Course: GPU Programming in Computer Vision" at Technische Universität München (summer semester 2017).  

Three MSc Computational Science & Engineering students contributed to this work (in alphabetical order):

* Apoorv Gupta
* Jorge Salazar 
* Jiho Yang

Four directories are present in this repository: CPU, GPU, IBFS, and Graphs

* CPU : Contains sequential version of the code
* GPU : Contains CUDA implemented GPU-parallelised version of the code
* IBFS : Contains IBFS (Incremental Breadth First Search Algorithm) code, which was served as a benchmark solver (http://www.cs.tau.ac.il/~sagihed/ibfs/index.html)
* graph : Contains some of test graphs in .bk file format. These test cases were taken from: http://www.cs.tau.ac.il/~sagihed/ibfs/benchmark.html

##### Compilation

Use Makefile

#### GPU Compilation 

From the cmake_build folder do:

cmake ..
make

#### Running CPU code 

After compiling do:

./sim <file.bk>

#### Running GPU code 

After compiling do:
./main <file.bk> -alpha <value> -rho <value> -it <value>

where "alpha" and "rho" are hyperparameters for computing time steps, and "it" is the maximum number of iterations

#### Bash script

./script
