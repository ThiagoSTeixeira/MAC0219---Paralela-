OUTPUT=mandelbrot
SRC_DIR=src/
IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11 -lm

NCC=nvcc
NFLAGS=-gencode arch=compute_50,code=[sm_50,compute_50] -Wno-deprecated-gpu-targets

MCC=mpicc
MFLAGS=-lm

CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda $(OUTPUT)_ompi

$(OUTPUT)_omp: $(SRC_DIR)$(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(SRC_DIR)$(OUTPUT)_omp.c

$(OUTPUT)_pth: $(SRC_DIR)$(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(SRC_DIR)$(OUTPUT)_pth.c

$(OUTPUT)_seq: $(SRC_DIR)$(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(SRC_DIR)$(OUTPUT)_seq.c

$(OUTPUT)_cuda: $(SRC_DIR)$(OUTPUT)_cuda.cu
	$(NCC) -o $(OUTPUT)_cuda $(NFLAGS) $(SRC_DIR)$(OUTPUT)_cuda.cu

$(OUTPUT)_ompi: $(SRC_DIR)$(OUTPUT)_ompi.c
	$(MCC) -o $(OUTPUT)_ompi $(MFLAGS) $(SRC_DIR)$(OUTPUT)_ompi.c

.PHONY: clean
clean:
	rm $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_cuda $(OUTPUT)_ompi *$(IMAGE)
