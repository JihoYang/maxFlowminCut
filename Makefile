#Include files
SOURCES=main.cpp primal_dual.cpp read_bk.cpp

#Compiler
#--------
CC = g++
CFLAGS = -std=c++11 -fstrict-overflow -Werror -Wshadow -Wstrict-overflow=4 -pedantic

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
