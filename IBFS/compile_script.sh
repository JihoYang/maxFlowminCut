#!/bin/bash

g++ -O3 -Wall -c -o main.o "main.cpp" 
g++ -O3 -Wall -c -o ibfs.o "ibfs.cpp" 
g++ -static-libgcc -static-libstdc++ -o IBFS main.o ibfs.o 
