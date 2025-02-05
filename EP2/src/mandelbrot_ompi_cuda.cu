#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "mpi.h"

#define min(x, y) (x < y ? x : y);
#define MASTER 0

// #define DEBUG 1
#define DEBUG 0

typedef struct process_args
{
    int start_y;
    int end_y;
    int start_x;
    int end_x;
} process_args;

struct timer_info
{
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
unsigned char **image_buffer;

int i_x_max;
int i_y_max;
int image_buffer_size;

int block_dim_x;
int block_dim_y;

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

void allocate_image_buffer()
{
    int rgb_size = 3;
    image_buffer =
        (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);

    for (int i = 0; i < image_buffer_size; i++)
    {
        image_buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
    };
};

void free_image_buffer()
{
    for (int i = 0; i < image_buffer_size; i++)
        free(image_buffer[i]);
    free(image_buffer);
}

void init(int argc, char *argv[], int rank)
{
    if (argc < 8)
    {
        if (rank == MASTER)
        {
            printf("usage:    mpirun -np [num_processes] --host localhost:[num_processes]"
                   " mandelbrot_ompi_cuda c_x_min c_x_max c_y_min c_y_max image_size"
                   " grid_x grid_y\n");
            printf("examples with image_size = 11500:\n");
            printf("    Full Picture:         mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_cuda -2.5 1.5 -2.0 2.0 11500 4 4\n");
            printf("    Seahorse Valley:      mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_cuda  -0.8 -0.7 0.05 0.15 11500 4 4\n");
            printf("    Elephant Valley:      mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_cuda  0.175 0.375 -0.1 0.1 11500 4 4\n");
            printf("    Triple Spiral Valley: mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_cuda  -0.188 -0.012 0.554 0.754 11500 4 4\n");
        }
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
    }
};

void update_rgb_buffer(int iteration, int x, int y)
{
    int color;

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

void write_to_file()
{
    FILE *file;
    char *filename = "output.ppm";
    char *comment = "# ";

    int max_color_component_value = 255;

    file = fopen(filename, "wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
            max_color_component_value);

    for (int i = 0; i < image_buffer_size; i++)
    {
        fwrite(image_buffer[i], 1, 3, file);
    };

    fclose(file);
};


void init_ompi_data(struct process_args *p_data, int n_process)
{
    int IMAGE_SIZE = image_size;
    int vertical_chunk_size, lin;

    vertical_chunk_size = IMAGE_SIZE / n_process;

    if (DEBUG)
        printf("[MASTER]: vertical chunk size : %d\n", vertical_chunk_size);

    lin = 0;
    for (int process_rank = 0; process_rank < n_process; process_rank++)
    {
        p_data[process_rank].start_x = 0;
        p_data[process_rank].end_x = IMAGE_SIZE;
        p_data[process_rank].start_y = lin;

        lin = process_rank == n_process - 1 ? IMAGE_SIZE : lin + vertical_chunk_size;
        p_data[process_rank].end_y = lin;

        if (DEBUG)
            printf("[MASTER]%d: %d %d %d %d\n", process_rank, p_data[process_rank].start_x,
                   p_data[process_rank].end_x, p_data[process_rank].start_y, p_data[process_rank].end_y);
    }
}

//

// MPI_Send(result, counter, MPI_INT, MASTER, 0, MPI_COMM_WORLD);


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


__global__ void compute_mandelbrot(int start_x, int end_x, int start_y, int end_y, int*result,
    double c_x_min, double c_y_min, double pixel_width, double pixel_height){

    int i_x;
    int i_y;
    int iteration;
    int pos;

    i_x = start_x + blockIdx.x*blockDim.x+threadIdx.x;
    i_y = start_y + blockIdx.y*blockDim.y+threadIdx.y;


    double c_y = c_y_min + i_y * pixel_height;

    if(fabs(c_y) < pixel_height / 2){
                c_y = 0.0;
    };

    double c_x = c_x_min + i_x * pixel_width;

    iteration = mandelbrot(c_x, c_y);

    pos = 3 * ((i_y - start_y) * (end_x - start_x) + i_x - start_x);

    result[pos] = iteration;
    result[pos + 1] = i_x;
    result[pos + 2] = i_y;

}

void cuda_compute_mandelbrot(process_args *process_data, int **result, int *result_size) {
    int start_x, start_y, end_x, end_y;
    int *dev_result;
    start_y = process_data->start_y;
    end_y = process_data->end_y;
    start_x = process_data->start_x;
    end_x = process_data->end_x;

    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid((int)ceil((end_x-start_x)/dimBlock.x),
                    (int)ceil((end_y-start_y)/dimBlock.y));

    *result_size = 3 * (end_x - start_x) * (end_y - start_y);

    *result = (int *)malloc( (*result_size) * sizeof(int));
    cudaMalloc((void**)&dev_result, (*result_size) * sizeof(int));

    compute_mandelbrot<<<dimGrid,dimBlock>>>(start_x, end_x, start_y,
        end_y, dev_result, c_x_min, c_y_min, pixel_width, pixel_height);

    cudaMemcpy(*result, dev_result, (*result_size) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_result);

}

void compute_mandelbrot_ompi(int argc, char *argv[], int num_processes, int rank_process)
{

    const int nitems = 4;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_process_data_type;
    MPI_Aint offsets[4];
    offsets[0] = offsetof(process_args, start_x);
    offsets[1] = offsetof(process_args, end_x);
    offsets[2] = offsetof(process_args, start_y);
    offsets[3] = offsetof(process_args, end_y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                           &mpi_process_data_type);
    MPI_Type_commit(&mpi_process_data_type);

    process_args *processes_args = NULL;

    if (rank_process == MASTER)
    {

        /* Process 0 will be a master process. It defines the ammount of work
        each process needs to execute and then sends the data that each process
        will work on*/

        processes_args = (process_args *) malloc(num_processes * sizeof(process_args));
        init_ompi_data(processes_args, num_processes);

        for (int p = 1; p < num_processes; p++)
            MPI_Send(&processes_args[p], 1, mpi_process_data_type, p, 0,
                     MPI_COMM_WORLD);
    }
    else
    {
        if (DEBUG)
            printf("[%d]: initiated\n", rank_process);
            
        int *result, result_size;
        process_args *process_data = (process_args *) malloc(sizeof(process_args));

        MPI_Recv(process_data, 1, mpi_process_data_type, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        if (DEBUG)
            printf("[%d]: received data\n", rank_process);
    
        cuda_compute_mandelbrot(process_data, &result, &result_size);

        MPI_Send(result, result_size, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

        if (DEBUG)
            printf("[%d]: finished computation\n", rank_process);

        free(process_data);
        free(result);
    }

    if (rank_process == MASTER)
    {
        int counters[num_processes];
        int *results[num_processes];

        for (int p = 1; p < num_processes; p++)
        {
            MPI_Status status;
            int count;
            MPI_Probe(p, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &count);

            if (DEBUG)
                printf("[MASTER]: process %d had count %d\n", p, count);

            int *buffer = (int *)malloc(count * sizeof(int));
            MPI_Recv(buffer, count, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            counters[p] = count;
            results[p] = buffer;
        }
        if (DEBUG)
            printf("[MASTER]: finished collecting results\n");

        allocate_image_buffer();

        if (DEBUG)
            printf("[MASTER]: image buffer allocated\n");

        process_args *master_data = &processes_args[MASTER];
        int *result, result_size;

        cuda_compute_mandelbrot(master_data, &result, &result_size);

        counters[MASTER] = result_size;
        results[MASTER] = result;

        for (int p = 0; p < num_processes; p++)
        {
            if (DEBUG)
                printf("[MASTER]: updatig values from %d\n", p);
            for (int i = 0; i < counters[p]; i = i + 3)
            {
                update_rgb_buffer(results[p][i], results[p][i + 1], results[p][i + 2]);
            }
            if (DEBUG)
                printf("[MASTER]: finished reading process %d results\n", p);
        }
        write_to_file();
        if (DEBUG)
            printf("[MASTER]: finished writing imagefile\n");
        
            free(processes_args);
            free_image_buffer();
            for (int p = 0; p < num_processes; p++)
                free(results[p]);
    }

    MPI_Finalize();
}

int main(int argc, char *argv[])
{

    int num_processes, rank_process;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_process);

    if (rank_process == MASTER)
    {
        timer.c_start = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
        gettimeofday(&timer.v_start, NULL);
    }

    init(argc, argv, rank_process);
    compute_mandelbrot_ompi(argc, argv, num_processes, rank_process);

    if (rank_process == MASTER)
    {
        timer.c_end = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
        gettimeofday(&timer.v_end, NULL);

        printf("%f\n",
               (double)(timer.t_end.tv_sec - timer.t_start.tv_sec) +
                   (double)(timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);
    }

    return 0;
};
