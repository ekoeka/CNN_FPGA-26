CXX = g++
CXXFLAGS = -O2 -Wall

all: sim_cnn

sim_cnn: tb_cnn.cpp cnn_hls.cpp weights.h
	$(CXX) $(CXXFLAGS) tb_cnn.cpp cnn_hls.cpp -o sim_cnn

run: sim_cnn
	./sim_cnn

clean:
	rm -f sim_cnn *.o
