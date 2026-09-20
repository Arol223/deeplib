CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g 

SRC = $(wildcard src/*.cpp)
OUT = deeplib

all:
	$(CXX) $(CXXFLAGS) -Iinclude $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)

debug:
	$(CXX) $(CXXFLAGS) -fsanitize=address,undefined -Iinclude $(SRC) -o $(OUT)

LIB_SRC = $(filter-out src/main.cpp, $(wildcard src/*.cpp))

test:
	$(CXX) $(CXXFLAGS) -Iinclude $(LIB_SRC) tests/test_main.cpp -o run_tests