#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;

int image_size;
unsigned char **image_buffer;
unsigned char **dev_image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;

int gradient_size = 16;
int colors[17][3] = {
    {66, 30, 15},
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3},
    {16, 16, 16},
};

int **dev_colors;

void allocate_image_buffer()
{
	int rgb_size = 3;
    image_buffer = (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);
    cudaMalloc((unsigned char **)&dev_image_buffer, sizeof(unsigned char *) * image_buffer_size);

    for (int i = 0; i < image_buffer_size; i++)
    {
        image_buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
        cudaMalloc((unsigned char *)&dev_image_buffer[i], sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[])
{
    if (argc < 6)
    {
        printf("usage: ./mandelbrot_seq c_x_min c_x_max c_y_min c_y_max image_size\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_seq -2.5 1.5 -2.0 2.0 11500\n");
        printf("    Seahorse Valley:      ./mandelbrot_seq -0.8 -0.7 0.05 0.15 11500\n");
        printf("    Elephant Valley:      ./mandelbrot_seq 0.175 0.375 -0.1 0.1 11500\n");
        printf("    Triple Spiral Valley: ./mandelbrot_seq -0.188 -0.012 0.554 0.754 11500\n");
        exit(0);
    }
    else
    {
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);

        i_x_max = image_size;
        i_y_max = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width = (c_x_max - c_x_min) / i_x_max;
        pixel_height = (c_y_max - c_y_min) / i_y_max;
    };
};

void init_colors(){

	cudaMalloc((int**)&dev_colors, sizeof(int *) * 17);
    for (int i = 0; i < 17; i++)
    {
    	cudaMalloc((int *)&dev_colors[i], sizeof(int) * 3);
    }

    cudaMemcpy(dev_colors, colors, sizeof(int *) * 17, cudaMemcpyHostToDevice);
}


void write_to_file()
{
    FILE *file;
    char *filename = "output.ppm";
    char *comment = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for (int i = 0; i < image_buffer_size; i++)
    {
        fwrite(image_buffer[i], 1, 3, file);
    };

    fclose(file);
};

__device__ void update_rgb_buffer(int iteration, int iteration_max, int x, int y, int i_y_max, unsigned char **image_buffer, int **colors)
{
    int color;
    int gradient_size = 16;

    if (iteration == iteration_max)
    {
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else
    {
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};


__device__ int mandelbrot(double c_x, double c_y, int iteration_max) {
    double z_x = 0;
    double z_y = 0;
    double z_x_squared = 0;
    double z_y_squared = 0;
    double escape_radius_squared = 4;

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

__global__ void compute_mandelbrot(unsigned char **image_buffer, double c_x_min, double c_y_min, double pixel_width, 
                                    double pixel_height, int iteration_max, int i_y_max, int image_size, int **colors){

    int dim = image_size;

    int pix_per_thread = dim * dim / (gridDim.x * blockDim.x);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int movement = pix_per_thread * thread_id;

    int iteration;

    for (int i = movement; i < movement + pix_per_thread; i++){
        int i_x = i % dim;
        int i_y = i / dim;
        double c_x = c_x_min + i_x * pixel_width;
        double c_y = c_y_min + i_y * pixel_height;

        iteration = mandelbrot(c_x, c_y, iteration_max);
        update_rgb_buffer(iteration, iteration_max, i_x, i_y, i_y_max, image_buffer, colors);

    }

    if (gridDim.x * blockDim.x * pix_per_thread < dim * dim
            && thread_id < (dim * dim) - (blockDim.x * gridDim.x)){
        int i = blockDim.x * gridDim.x * pix_per_thread + thread_id;
        int i_x = i % dim;
        int i_y = i / dim;
        double c_x = c_x_min + i_x * pixel_width;
        double c_y = c_y_min + i_y * pixel_height;

        iteration = mandelbrot(c_x, c_y, iteration_max);
        update_rgb_buffer(iteration, iteration_max, i_x, i_y, i_y_max, image_buffer, colors);
    }
}

int main(int argc, char *argv[])
{
    init(argc, argv);

    allocate_image_buffer();

    init_colors();

    dim3 numBlocks(image_size, image_size);

    compute_mandelbrot <<<image_size,image_size>>>(dev_image_buffer, c_x_min, c_y_min, pixel_width, pixel_height, iteration_max, i_y_max, image_size, dev_colors);

    cudaMemcpy(image_buffer, dev_image_buffer, sizeof(unsigned char *) * image_buffer_size, cudaMemcpyDeviceToHost);

    write_to_file();

    return 0;
};