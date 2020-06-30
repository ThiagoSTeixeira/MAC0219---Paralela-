
OUTPUT=mandelbrot_ompi
MCC=mpicc --showme

.PHONY: all

all: $(OUTPUT)

$(OUTPUT): $(OUTPUT).c
	$(MCC) -o $(OUTPUT) $(OUTPUT).c

.PHONY: clean
clean:
	rm $(OUTPUT)