
pr:
	@echo "Make ....."
	nvcc -std=c++11 properties.cu -o properties
	./properties
	rm -r properties
	
cpu:
	g++ juliaCPU.cpp -std=c++11 -O3 lodepng.cpp -o jCPU
	./jCPU -0.8 0.2 juliaCPU1.png
	./jCPU 0 0.8 juliaCPU2.png
	rm -r jCPU
cuda:
	nvcc -std=c++11 juliaCUDA.cu lodepng.cpp -o jCUDA
	./jCUDA -0.8 0.2 juliaCUDA1.png
	./jCUDA 0 0.8 juliaCUDA2.png
	rm -r jCUDA
