#Include files
SOURCES=main.cpp primal_dual.cpp mathOperations.cpp read_bk.cpp postProcessing.cpp

#Compiler
#--------
CC = g++
CFLAGS = -std=c++11 -fstrict-overflow -Werror -Wshadow -Wstrict-overflow=4 -Wno-write-strings 

#Linker flags
#------------
LDFLAGS= -lm

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=sim

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(OBJECTS): %.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
