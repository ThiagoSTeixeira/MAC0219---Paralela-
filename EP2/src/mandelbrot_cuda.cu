//Compilar com: nvcc -gencode arch=compute_50,code=[sm_50,compute_50] mandelbrot_cuda.cu -o mandelbrot -Wno-deprecated-gpu-targets

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;


double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
unsigned char *image_buffer_red;
unsigned char *image_buffer_green;
unsigned char *image_buffer_blue;
unsigned char *dev_image_buffer_red;
unsigned char *dev_image_buffer_blue;
unsigned char *dev_image_buffer_green;

unsigned char **pixels;

int block_dim_x;
int block_dim_y;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
int color_red[17]   = {66, 25, 9, 4, 0, 12, 24, 57, 134, 211, 241, 248, 255, 204, 153, 106, 16};
int color_green[17] = {30, 7, 1, 4, 7, 44, 82, 125, 181, 236, 233, 201, 170, 128, 87, 52, 16};
int color_blue[17]  = {15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191, 95, 0, 0, 0, 3, 16};
int *dev_color_red;
int *dev_color_green;
int *dev_color_blue;

void allocate_image_buffer()
{
	//int rgb_size = 3;
    image_buffer_red = (unsigned char *)malloc(sizeof(unsigned char) * image_buffer_size);
    cudaMalloc((void**)&dev_image_buffer_red, image_buffer_size * sizeof(unsigned char));

    image_buffer_green = (unsigned char *)malloc(sizeof(unsigned char) * image_buffer_size);
    cudaMalloc((void**)&dev_image_buffer_green, image_buffer_size * sizeof(unsigned char));

    image_buffer_blue = (unsigned char *)malloc(sizeof(unsigned char) * image_buffer_size);
    cudaMalloc((void**)&dev_image_buffer_blue, image_buffer_size * sizeof(unsigned char));
};

void init(int argc, char *argv[])
{
    if (argc < 8)
    {
        printf("usage: ./mandelbrot_seq c_x_min c_x_max c_y_min c_y_max image_size dimX dimY\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_seq -2.5 1.5 -2.0 2.0 4096 8 64\n");
        printf("    Seahorse Valley:      ./mandelbrot_seq -0.8 -0.7 0.05 0.15 4096 32 32\n");
        printf("    Elephant Valley:      ./mandelbrot_seq 0.175 0.375 -0.1 0.1 4096 16 64\n");
        printf("    Triple Spiral Valley: ./mandelbrot_seq -0.188 -0.012 0.554 0.754 4096 1 32\n");
        exit(0);
    }
    else
    {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &block_dim_x);
        sscanf(argv[7], "%d", &block_dim_y);

        i_x_max = image_size;
        i_y_max = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width = (c_x_max - c_x_min) / i_x_max;
        pixel_height = (c_y_max - c_y_min) / i_y_max;
    };
};

void init_colors(){

    int color_size = 17 * sizeof(int);
    cudaMalloc((void**)&dev_color_red, color_size);
    cudaMalloc((void**)&dev_color_green, color_size);
    cudaMalloc((void**)&dev_color_blue, color_size);

    cudaMemcpy(dev_color_red, color_red, color_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color_green, color_green, color_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color_blue, color_blue, color_size, cudaMemcpyHostToDevice);
}

void allocate_pixels(){
    int rgb_size = 3;
    pixels = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for(int i = 0; i < image_buffer_size; i++){
        pixels[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
}

void set_pixels(){
    for(int i = 0; i < image_buffer_size; i++){
        pixels[i][0] = image_buffer_red[i];
        pixels[i][1] = image_buffer_green[i];
        pixels[i][2] = image_buffer_blue[i];
    }
}

void write_to_file()
{
    FILE *file;
    const char *filename = "output.ppm";
    const char *comment = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);


    for(int i = 0; i < image_buffer_size; i++){
        fwrite(pixels[i], 1 , 3, file);
    };

    fclose(file);
};

__device__ void update_rgb_buffer(int iteration, int x, int y, int image_size, 
                                  unsigned char *image_buffer_red, unsigned char *image_buffer_green, unsigned char *image_buffer_blue,
                                  int *color_red, int *color_green, int *color_blue)
{

    int gradient_size = 16;
    int iteration_max = 200;
    int color;

    if (iteration == iteration_max)
    {
        image_buffer_red[(image_size * y) + x] = color_red[gradient_size];
        image_buffer_green[(image_size * y) + x] = color_green[gradient_size];
        image_buffer_blue[(image_size * y) + x] = color_blue[gradient_size];
    }

    else
    {
        color = iteration % gradient_size;

        image_buffer_red[(image_size * y) + x] = color_red[color];
        image_buffer_green[(image_size * y) + x] = color_green[color];
        image_buffer_blue[(image_size * y) + x] = color_blue[color];
    };
};


__device__ int mandelbrot(double c_x, double c_y) {
    double z_x = 0;
    double z_y = 0;
    double z_x_squared = 0;
    double z_y_squared = 0;
    double escape_radius_squared = 4;
    int iteration_max = 200;

    int iteration;

    for (iteration = 0;
         iteration < iteration_max &&
         ((z_x_squared + z_y_squared) < escape_radius_squared);
         iteration++) {
                z_y = 2 * z_x * z_y + c_y;
                z_x = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
    };

    return iteration;
}



__global__ void compute_mandelbrot(unsigned char *image_buffer_red, unsigned char *image_buffer_green, unsigned char *image_buffer_blue, 
                                    double c_x_min, double c_y_min, double pixel_width, double pixel_height, int image_size,
                                    int *color_red, int *color_green, int *color_blue){

    int i_x;
    int i_y;
    int iteration;


    i_x = blockIdx.x*blockDim.x+threadIdx.x;
    i_y = blockIdx.y*blockDim.y+threadIdx.y;


    double c_y = c_y_min + i_y * pixel_height;

    if(fabs(c_y) < pixel_height / 2){
                c_y = 0.0;
    };

    double c_x = c_x_min + i_x * pixel_width;

    iteration = mandelbrot(c_x, c_y);
    update_rgb_buffer(iteration, i_x, i_y, image_size, image_buffer_red, image_buffer_green, image_buffer_blue, color_red, color_green, color_blue);
}

int main(int argc, char *argv[])
{
    init(argc, argv);

    allocate_image_buffer();

    init_colors();

    int time;


    dim3 dimBlock(block_dim_x, block_dim_y);
	dim3 dimGrid((int)ceil(image_size/dimBlock.x),(int)ceil(image_size/dimBlock.y));

	//MEDICAO DE TEMPO
	timer.c_start = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
    gettimeofday(&timer.v_start, NULL);

    compute_mandelbrot <<<dimGrid, dimBlock>>>(dev_image_buffer_red, dev_image_buffer_green, dev_image_buffer_blue, 
                                                    c_x_min, c_y_min, pixel_width, pixel_height, image_size,
                                                    dev_color_red, dev_color_green, dev_color_blue);

    cudaMemcpy(image_buffer_red, dev_image_buffer_red, sizeof(unsigned char) * image_buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(image_buffer_green, dev_image_buffer_green, sizeof(unsigned char) * image_buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(image_buffer_blue, dev_image_buffer_blue, sizeof(unsigned char) * image_buffer_size, cudaMemcpyDeviceToHost);

    timer.c_end = clock();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
    gettimeofday(&timer.v_end, NULL);
    //FIM DA MEDICAO

    cudaFree(dev_color_red);
    cudaFree(dev_color_green);
    cudaFree(dev_color_blue);

    allocate_pixels();
    set_pixels();

    cudaFree(dev_image_buffer_red);
    cudaFree(dev_image_buffer_green);
    cudaFree(dev_image_buffer_blue);

    write_to_file();

    printf("%f\n",
    (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
    (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);

    time = (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
    (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0;

    return time;
};
