#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <time.h>
#include <sys/time.h>

#include "mpi.h"

#define min(x, y) (x < y ? x : y);
#define MASTER 0

#define DEBUG 0

struct process_args
{
    int start_y;
    int end_y;
    int start_x;
    int end_x;
};

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
int n_threads;

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

void init(int argc, char *argv[], int rank)
{
    if (argc < 7)
    {
        if (rank == MASTER)
        {
            printf("usage:    mpirun -np [num_processes] --host "
                   "localhost:[num_processes] mandelbrot_ompi_omp c_x_min c_x_max "
                   "c_y_min c_y_max image_size n_threads\n");
            printf("examples with image_size = 11500:\n");
            printf("    Full Picture:         mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_omp -2.5 1.5 -2.0 2.0 11500 4\n");
            printf("    Seahorse Valley:      mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_omp  -0.8 -0.7 0.05 0.15 11500 4\n");
            printf("    Elephant Valley:      mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_omp  0.175 0.375 -0.1 0.1 11500 4\n");
            printf("    Triple Spiral Valley: mpirun -np 9 --host localhost:9 "
                   "mandelbrot_ompi_omp  -0.188 -0.012 0.554 0.754 11500 4\n");
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
        sscanf(argv[6], "%d", &n_threads);

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

int *compute_mandelbrot(struct process_args *process_data, int rank)
{
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;
    int i_x;
    int i_y;
    double c_x;
    double c_y;

    int start_y = process_data->start_y;
    int end_y = process_data->end_y;
    int start_x = process_data->start_x;
    int end_x = process_data->end_x;
    int counter;
    int *result;
    int pos;

    if (end_y < start_y || end_x < start_x)
        counter = 0;
    else
        counter = 3 * (end_x - start_x) * (end_y - start_y);

    result = (int *)malloc(counter * sizeof(int));

    if (DEBUG)
        printf("[%d]: %d %d %d %d\n", rank, start_x, end_x, start_y, end_y);

#pragma omp parallel for private(c_y) schedule(dynamic) num_threads(n_threads)
    for (i_y = start_y; i_y < end_y; i_y++)
    {
        c_y = c_y_min + i_y * pixel_height;

        if (fabs(c_y) < pixel_height / 2)
        {
            c_y = 0.0;
        };

#pragma omp parallel for private(z_x, z_y, z_x_squared, z_y_squared, c_x, iteration, pos) schedule(dynamic) num_threads(n_threads)
        for (i_x = start_x; i_x < end_x; i_x++)
        {
            c_x = c_x_min + i_x * pixel_width;

            z_x = 0.0;
            z_y = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;

            for (iteration = 0; iteration < iteration_max &&
                                ((z_x_squared + z_y_squared) < escape_radius_squared);
                 iteration++)
            {
                z_y = 2 * z_x * z_y + c_y;
                z_x = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };
            pos = 3 * ((i_y - start_y) * (end_x - start_x) + i_x - start_x);

            if (DEBUG && pos > counter)
                printf("[%d]: pos = %d counter = %d\n", rank, pos, counter);

            result[pos] = iteration;
            result[pos + 1] = i_x;
            result[pos + 2] = i_y;
        };
    };

    MPI_Send(result, counter, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
};

void init_ompi_data(struct process_args *t_data, int n_process)
{
    long t = 0;
    int IMAGE_SIZE = image_size;
    int ver_quadrant_size, hor_quadrant_size, lin, col;

    ver_quadrant_size = (IMAGE_SIZE / (int)sqrt(n_process)) + 1;
    hor_quadrant_size = (IMAGE_SIZE / (int)sqrt(n_process)) + 1;

    if (DEBUG)
    {
        printf("[MASTER]: vertical quadrant value : %d\n", ver_quadrant_size);
        printf("[MASTER]: horizontal quadrant value : %d\n", hor_quadrant_size);
    }
    for (lin = 0; lin < IMAGE_SIZE; lin += ver_quadrant_size)
    {
        for (col = 0; col < IMAGE_SIZE; col += hor_quadrant_size)
        {
            if (t == n_process - 2) // last process takes the rest of the pixels
            {
                t_data[t].start_x = col;
                t_data[t].end_x = IMAGE_SIZE;
                t_data[t].start_y = lin;
                t_data[t].end_y = IMAGE_SIZE;
                if (DEBUG)
                    printf("[MASTER]%d: %d %d %d %d\n", t + 1, t_data[t].start_x, t_data[t].end_x,
                           t_data[t].start_y, t_data[t].end_y);
                return;
            }
            t_data[t].start_x = col;
            t_data[t].end_x = min(col + hor_quadrant_size, IMAGE_SIZE);
            t_data[t].start_y = lin;
            t_data[t].end_y = min(lin + ver_quadrant_size, IMAGE_SIZE);
            if (DEBUG)
                printf("[MASTER]%d: %d %d %d %d\n", t + 1, t_data[t].start_x, t_data[t].end_x,
                       t_data[t].start_y, t_data[t].end_y);
            t += 1;
        }
    }

    // Leftover processes
    for (int i = t; i < n_process; i++)
    {
        t_data[i].start_x = 0;
        t_data[i].end_x = -1;
        t_data[i].start_y = 0;
        t_data[i].end_y = -1;
    }
}

void compute_mandelbrot_ompi(int argc, char *argv[], int num_processes, int rank_process)
{

    const int nitems = 4;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_process_data_type;
    MPI_Aint offsets[4];
    offsets[1] = offsetof(struct process_args, start_y);
    offsets[3] = offsetof(struct process_args, end_y);
    offsets[0] = offsetof(struct process_args, start_x);
    offsets[2] = offsetof(struct process_args, end_x);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                           &mpi_process_data_type);
    MPI_Type_commit(&mpi_process_data_type);

    if (rank_process == MASTER)
    {
        struct process_args *processes_data = NULL;

        /* Process 0 will be a master process. It defines the ammount of work
        each process needs to execute and then sends the data that each process
        will work on*/

        processes_data = malloc(num_processes * sizeof(struct process_args));
        init_ompi_data(processes_data, num_processes);

        for (int p = 0; p < num_processes - 1; p++)
            MPI_Send(&processes_data[p], 1, mpi_process_data_type, p + 1, 0,
                     MPI_COMM_WORLD);
    }
    else
    {
        if (DEBUG)
            printf("[%d]: initiated\n", rank_process);

        struct process_args *process_data = malloc(sizeof(struct process_args));
        MPI_Recv(process_data, 1, mpi_process_data_type, MASTER, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        if (DEBUG)
            printf("[%d]: received data\n", rank_process);

        compute_mandelbrot(process_data, rank_process);

        if (DEBUG)
            printf("[%d]: finished computation\n", rank_process);
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

            if (DEBUG)
                printf("[MASTER]: probe [%d] completed with error %d\n", p, status.MPI_ERROR);

            MPI_Get_count(&status, MPI_INT, &count);

            if (DEBUG)
            {
                printf("[MASTER]: get_count [%d] completed with error %d\n", p, status.MPI_ERROR);
                printf("[MASTER]: process %d had count %d\n", p, count);
            }

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

        for (int p = 1; p < num_processes; p++)
        {
            if (DEBUG)
            {
                printf("[MASTER]: updating values from %d\n", p);
                printf("[MASTER]: counters[%d] = %d / mod3: %d\n", p, counters[p], counters[p] % 3);
            }

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
