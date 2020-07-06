
OUTPUT=mandelbrot_ompi
MCC=mpicc
MFLAGS=-lm

.PHONY: all
all: $(OUTPUT)

$(OUTPUT): $(OUTPUT).c
	$(MCC) -o $(OUTPUT) $(OUTPUT).c $(MFLAGS)

.PHONY: clean
clean:
	rm $(OUTPUT)