CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -g -I/usr/include/eigen3

SRC = $(wildcard src/*.cpp)
OUT = deeplib

.PHONY: all test debug clean test-tsan

all:
	$(CXX) $(CXXFLAGS) -Iinclude $(SRC) -o $(OUT) -fopt-info-vec 2>&1 | grep ops.cpp

clean:
	rm -f $(OUT)

debug:
	$(CXX) $(CXXFLAGS) -fsanitize=address,undefined -DDEEPLIB_POISON -Iinclude $(SRC) -o $(OUT)

LIB_SRC = $(filter-out src/main.cpp, $(wildcard src/*.cpp))

test:
	$(CXX) $(CXXFLAGS) -Iinclude $(LIB_SRC) -DDEEPLIB_POISON tests/test_main.cpp -o run_tests

test-tsan:
	$(CXX) $(CXXFLAGS) -O1 -fsanitize=thread -Iinclude $(LIB_SRC) -DDEEPLIB_POISON tests/test_main.cpp -o run_tests_tsan