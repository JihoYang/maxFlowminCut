# maxFlowminCut

### GPU accelerated first order primal-dual algorithm for solving convex optimization problems, and its application in maximum flow min cut graph problem

#### CPU Compilation 

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
