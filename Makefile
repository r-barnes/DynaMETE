all:
	$(CXX) -O3 -g -march=native dynamete.cpp -Wall -ffast-math -fopenmp -lboost_iostreams