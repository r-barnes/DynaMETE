all:
	$(CXX) -O3 -g -mcpu=native -o dynamete.exe dynamete.cpp -Wall -ffast-math -fopenmp -Wno-unknown-pragmas #-fsanitize=address

summit:
	pgc++ -fast -acc -std=c++11 -o dynamete.exe -ta=tesla,pinned,cc60 -Minfo=accel dynamete.cpp 