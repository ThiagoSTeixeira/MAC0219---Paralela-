#File constants
OUTPUT=mandelbrot
SRC=src/mandelbrot
IMAGE=.ppm
COMP_FILE=EP2_MAC219.tar
DATA=.csv

#Compilers
CC=gcc
NCC=nvcc
MCC=mpicc

#Compress option
COMPRESS=tar -cvf

#Compilation Flags
CC_OPT=-std=c11 -lm
CC_OMP=-fopenmp
CC_PTH=-pthread
NCC_OPT=-gencode arch=compute_50,code=[sm_50,compute_50] -Wno-deprecated-gpu-targets
MCC_OPT=-lm
OMPICOMPATCC_OPT=-I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi
#Na rede linux:
# OMPICOMPATCC_OPT=-I/home/bcc/caiofcas/.local/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi


.PHONY: all
all: $(OUTPUT)_omp  	\
	 $(OUTPUT)_pth  	\
	 $(OUTPUT)_seq  	\
	 $(OUTPUT)_ompi 	\
	 $(OUTPUT)_ompi_omp \
	 $(OUTPUT)_cuda 	\
	 $(OUTPUT)_ompi_cuda

$(OUTPUT)_seq: $(SRC)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(SRC)_seq.c

$(OUTPUT)_omp: $(SRC)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(SRC)_omp.c

$(OUTPUT)_pth: $(SRC)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(SRC)_pth.c

$(OUTPUT)_ompi: $(SRC)_ompi.c
	$(MCC) -o $(OUTPUT)_ompi $(MCC_OPT) $(SRC)_ompi.c

$(OUTPUT)_ompi_omp: $(SRC)_ompi_omp.c
	$(MCC) -o $(OUTPUT)_ompi_omp $(MCC_OPT) $(CC_OMP) $(SRC)_ompi_omp.c

$(OUTPUT)_cuda: $(SRC)_cuda.cu
	$(NCC) -o $(OUTPUT)_cuda $(NCC_OPT) $(SRC)_cuda.cu

$(OUTPUT)_ompi_cuda: $(SRC)_ompi_cuda.cu
	$(NCC) -o $(OUTPUT)_ompi_cuda $(MCC_OPT) $(NCC_OPT) $(OMPICOMPATCC_OPT) $(SRC)_ompi_cuda.cu

.PHONY: clean
clean:
	rm $(OUTPUT)_*	\
	   *$(IMAGE)	\
	   $(COMP_FILE)

.PHONY: compress
compress:
	$(COMPRESS) $(COMP_FILE) ./src EP2.pdf RelatorioEP2.ipynb Makefile *$(DATA)