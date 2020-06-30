#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

#define min(x, y) (x < y ? x : y);

struct process_args {
  int start_y;
  int end_y;
  int start_x;
  int end_x;
  //  int rank;
};

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
    {66, 30, 15},    {25, 7, 26},     {9, 1, 47},      {4, 4, 73},
    {0, 7, 100},     {12, 44, 138},   {24, 82, 177},   {57, 125, 209},
    {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95},
    {255, 170, 0},   {204, 128, 0},   {153, 87, 0},    {106, 52, 3},
    {16, 16, 16},
};

void allocate_image_buffer() {
  int rgb_size = 3;
  image_buffer =
      (unsigned char **)malloc(sizeof(unsigned char *) * image_buffer_size);

  for (int i = 0; i < image_buffer_size; i++) {
    image_buffer[i] = (unsigned char *)malloc(sizeof(unsigned char) * rgb_size);
  };
};

void init() {
    /* Only computes Triple Spiral Valley with image size = 4096*/
    c_x_min = -0.188;
    c_x_max = -0.012;
    c_y_min = 0.554;
    c_y_max = 0.754;
    image_size = 4096;

    i_x_max = image_size;
    i_y_max = image_size;
    image_buffer_size = image_size * image_size;

    pixel_width = (c_x_max - c_x_min) / i_x_max;
    pixel_height = (c_y_max - c_y_min) / i_y_max;
};

void update_rgb_buffer(int iteration, int x, int y) {
  int color;

  if (iteration == iteration_max) {
    image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
    image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
    image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
  } else {
    color = iteration % gradient_size;

    image_buffer[(i_y_max * y) + x][0] = colors[color][0];
    image_buffer[(i_y_max * y) + x][1] = colors[color][1];
    image_buffer[(i_y_max * y) + x][2] = colors[color][2];
  };
};

void write_to_file() {
  FILE *file;
  char *filename = "output.ppm";
  char *comment = "# ";

  int max_color_component_value = 255;

  file = fopen(filename, "wb");

  fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment, i_x_max, i_y_max,
          max_color_component_value);

  for (int i = 0; i < image_buffer_size; i++) {
    fwrite(image_buffer[i], 1, 3, file);
  };

  fclose(file);
};

void compute_mandelbrot(struct process_args *process_data) {
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

  for (i_y = start_y; i_y < end_y; i_y++) {
    c_y = c_y_min + i_y * pixel_height;

    if (fabs(c_y) < pixel_height / 2) {
      c_y = 0.0;
    };

    for (i_x = start_x; i_x < end_x; i_x++) {
      c_x = c_x_min + i_x * pixel_width;

      z_x = 0.0;
      z_y = 0.0;

      z_x_squared = 0.0;
      z_y_squared = 0.0;

      for (iteration = 0; iteration < iteration_max &&
                          ((z_x_squared + z_y_squared) < escape_radius_squared);
           iteration++) {
        z_y = 2 * z_x * z_y + c_y;
        z_x = z_x_squared - z_y_squared + c_x;

        z_x_squared = z_x * z_x;
        z_y_squared = z_y * z_y;
      };

      int buffer[3] = {iteration, i_x, i_y};
      MPI_Send(&buffer, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
    };
  };
};

void init_ompi_data(struct process_args *t_data, int n_process) {
  long t = 0;
  int IMAGE_SIZE = image_size;
  int ver_quadrant_size, hor_quadrant_size, lin, col;

  ver_quadrant_size = IMAGE_SIZE / (int)sqrt(n_process);
  hor_quadrant_size = IMAGE_SIZE / (int)((double)n_process / sqrt(n_process));

  for (lin = 0; lin < IMAGE_SIZE; lin += ver_quadrant_size) {
    for (col = 0; col < IMAGE_SIZE; col += hor_quadrant_size) {
      t_data[t].start_x = col;
      t_data[t].end_x = min(col + hor_quadrant_size, IMAGE_SIZE);
      t_data[t].start_y = lin;
      t_data[t].end_y = min(lin + ver_quadrant_size, IMAGE_SIZE);
      t += 1;
      if (t == n_threads - 1)  // last process takes the rest of the pixels
      {
        t_data[t].start_x = col;
        t_data[t].end_x = IMAGE_SIZE;
        t_data[t].start_y = lin;
        t_data[t].end_y = min(lin + ver_quadrant_size, IMAGE_SIZE);
        return;
      }
    }
  }
}

// void compute_mandelbrot_threads() {
//   pthread_t *thread;
//   struct thread_args *thread_data;
//   pthread_attr_t attr;
//   int rc, err_code;
//   long t;
//   void *status;

//   thread = malloc(n_threads * sizeof(pthread_t));
//   thread_data = malloc(n_threads * sizeof(struct thread_args));

//   /* Initialize and set thread detached attribute */
//   pthread_attr_init(&attr);
//   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

//   init_thread_data(thread_data);

//   for (t = 0; t < n_threads; t++) {
//     rc = pthread_create(&thread[t], &attr, compute_mandelbrot,
//                         (void *)&thread_data[t]);
//     if (rc) {
//       printf("ERROR; return code from pthread_create() is %d\n", rc);
//       exit(-1);
//     }
//   }

//   pthread_attr_destroy(&attr);

//   // join loop
//   for (t = 0; t < n_threads; t++) {
//     err_code = pthread_join(thread[t], &status);
//     if (err_code) {
//       printf("ERROR; return code from ,pthread_join() is %d\n", err_code);
//       exit(-1);
//     };
//   };
// };

// void init_process_data(struct process_args *process_data, int rank) {}

void compute_mandelbrot_ompi(int argc, char *argv[]) {
  int num_processes, rank_process;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_process);

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

  if (rank_process == 0) {
    /* Process 0 will be a master process. It defines the ammount of work each process
    needs to execute and then sends the data that each process will work on*/
    init();
    allocate_image_buffer();

    struct process_args *process_data;
    process_data = malloc(num_processes * sizeof(struct process_args));
    init_ompi_data(process_data, num_processes);
    for (int p = 1; p < num_processes; p++) {
      MPI_Send(&process_data[p], 1, mpi_process_data_type, p, 0,
               MPI_COMM_WORLD);
    }

    int all_processes_finished = 0;
    MPI_Status status;
    while (all_processes_finished < num_processes - 1) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      // if tag value is 1 then some process have finished
      if (status.MPI_TAG == 1) {
        printf("Work done\n");
        int work_done;
        MPI_Recv(&work_done, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        all_processes_finished++;
      } else {
        int buffer[3];
        MPI_Recv(&buffer, 3, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        //printf("%d, %d, %d\n", buffer[0], buffer[1], buffer[2]);
        update_rgb_buffer(buffer[0], buffer[1], buffer[2]);
      }
    }
    printf('Vou escrever no arquivo!!!\n');
    write_to_file();

  } else {
    struct process_args *process_data = malloc(sizeof(struct process_args));
    MPI_Recv(process_data, 1, mpi_process_data_type, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    compute_mandelbrot(process_data);
    int work_done = 1;
    MPI_Send(&work_done, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}

int main(int argc, char *argv[]) {
  //printf("aaa\n");
  compute_mandelbrot_ompi(argc, argv);
  return 0;
};
