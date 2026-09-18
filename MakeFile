CXX = g++
CXXFLAGS = -std=c++20 -Wall -g

SRC = $(wildcard src/*.cpp)
OUT = deeplib

all:
	$(CXX) $(CXXFLAGS) -Iinclude $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)