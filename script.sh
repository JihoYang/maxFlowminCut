#!/bin/bash

alpha=1
rho=1
it=100000

cd GPU/cmake_build/
rm CMakeCache.txt

cmake ..

make

cd ../../

graph_name=graph3x3.max.bk

echo
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo ///////////////////////////////////////  $graph_name  /////////////////////////////////////
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo

echo .................. Starting IBFS code .................. 
echo

cd IBFS/

./IBFS ../graphs/$graph_name 
echo
echo .................. IBFS code ends ..................

echo
cd ../GPU/cmake_build/

echo .................. Starting GPU code ..................
echo
./main ../../graphs/$graph_name -alpha $alpha -rho $rho -it $it
echo
echo .................. GPU code ends ..................

echo .................. Showing how the graph looks like .............

python3.5 matrix_data.py

graph_name=car_16bins.bk

echo
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo ///////////////////////////////////////  $graph_name  /////////////////////////////////////
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo


echo .................. Starting IBFS code .................. 
echo

cd ../../IBFS/

./IBFS ../graphs/$graph_name 
echo
echo .................. IBFS code ends ..................

echo
cd ../GPU/cmake_build/

echo .................. Starting GPU code ..................
echo
./main ../../graphs/$graph_name -alpha $alpha -rho $rho -it $it
echo
echo .................. GPU code ends ..................

echo .................. Showing how the graph looks like .............

python3.5 matrix_data.py

graph_name=BVZ-venus0.bk

echo
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo ///////////////////////////////////////  $graph_name  /////////////////////////////////////
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo

echo .................. Starting IBFS code .................. 
echo
cd ../../IBFS/

./IBFS ../graphs/$graph_name 
echo
echo .................. IBFS code ends ..................

echo
cd ../GPU/cmake_build/

echo .................. Starting GPU code ..................
echo
./main ../../graphs/$graph_name -alpha $alpha -rho $rho -it $it
echo
echo .................. GPU code ends ..................

echo .................. Showing how the graph looks like .............

python3.5 matrix_data.py


graph_name=wide_graph.bk

echo
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo ///////////////////////////////////////  $graph_name  /////////////////////////////////////
echo ///////////////////////////////////////////////////////////////////////////////////////////
echo

echo .................. Starting IBFS code .................. 
echo
cd ../../IBFS/

./IBFS ../graphs/$graph_name 
echo
echo .................. IBFS code ends ..................

echo
cd ../GPU/cmake_build/

echo .................. Starting GPU code ..................
echo
./main ../../graphs/$graph_name -alpha $alpha -rho $rho -it $it
echo
echo .................. GPU code ends ..................

echo .................. Showing how the graph looks like .............

python3.5 matrix_data.py

cd ../../graphs
rm *.compiled










