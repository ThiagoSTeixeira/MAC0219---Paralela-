OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11 -lm

NCC=nvcc
NFLAGS=-gencode arch=compute_50,code=[sm_50,compute_50] -Wno-deprecated-gpu-targets

CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(OUTPUT)_omp.c

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(OUTPUT)_pth.c

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

$(OUTPUT)_cuda: $(OUTPUT)_cuda.cu
	$(NCC) -o $(OUTPUT)_cuda $(NFLAGS) $(OUTPUT)_cuda.cu

.PHONY: clean
clean:
	rm $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda *$(IMAGE)
