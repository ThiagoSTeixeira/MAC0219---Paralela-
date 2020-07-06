OUTPUT=mandelbrot
SRC=src/mandelbrot
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
all: $(OUTPUT)_omp  \
	 $(OUTPUT)_pth  \
	 $(OUTPUT)_seq  \
	 $(OUTPUT)_cuda \
	 $(OUTPUT)_ompi \
	 $(OUTPUT)_ompi_omp

$(OUTPUT)_omp: $(SRC)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(SRC)_omp.c

$(OUTPUT)_pth: $(SRC)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(SRC)_pth.c

$(OUTPUT)_seq: $(SRC)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(SRC)_seq.c

$(OUTPUT)_cuda: $(SRC)_cuda.cu
	$(NCC) -o $(OUTPUT)_cuda $(NFLAGS) $(SRC)_cuda.cu

$(OUTPUT)_ompi: $(SRC)_ompi.c
	$(MCC) -o $(OUTPUT)_ompi $(MFLAGS) $(SRC)_ompi.c

$(OUTPUT)_ompi_omp: $(SRC)_ompi_omp.c
	$(MCC) -o $(OUTPUT)_ompi_omp $(MFLAGS) $(CC_OPT) $(SRC)_ompi_omp.c


.PHONY: clean
clean:
	rm $(OUTPUT)_omp 		\
	   $(OUTPUT)_pth 		\
	   $(OUTPUT)_seq 		\
	   $(OUTPUT)_cuda 		\
	   $(OUTPUT)_ompi 		\
	   $(OUTPUT)_ompi_omp 	\
	   *$(IMAGE)			\
