# Reader of .bk files
The files [read_bk.h](https://github.com/jy1737/maxFlowminCut/blob/read_bk/read_file/read_bk.h) and [read_bk.cpp](https://github.com/jy1737/maxFlowminCut/blob/read_bk/read_file/read_bk.cpp) implement the .bk reader for this project.
To compile it use:
```
g++ -std=c++11 -c -o read_bk.o read_bk.cpp
g++ -std=c++11 -c -o main.o main.cpp
g++ -std=c++11 -o main main.o read_bk.o
```

You can test it with:
```
./main test.bk
```


## TODO's:
 - Write a makefile to compile all together.
 - Use sparse matrix format for matrix [w](https://github.com/jy1737/maxFlowminCut/blob/read_bk/read_file/read_bk.h#L19)
 - Look for CUDA libraries to read faster (if possible).
