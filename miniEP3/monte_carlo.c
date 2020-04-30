/*
| Nome | NUSP |
|------|------|
| Caio Andrade | 9797232 |
| Caio Fontes | 10692061 |
| Eduardo Laurentino | 8988212 |
| Thiago Teixeira | 10736987 |
| Washington Meireles | 10737157 |
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef VERBOSE
#define VERBOSE 0
#endif

#define FUNCTIONS 1

struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;

char *usage_message = "usage: ./monte_carlo SAMPLES FUNCTION_ID N_THREADS\n";

struct function {
    long double (*f)(long double);
    long double interval[2];
};

long double rand_interval[] = {0.0, (long double) RAND_MAX};

long double f1(long double x){
    return 2 / (sqrt(1 - (x * x)));
}

struct function functions[] = {
                               {&f1, {0.0, 1.0}}
};

// Your thread data structures go here

struct thread_data{
    int thread_id,
    	sample_begin,
    	sample_end;    
    long double *sample, *result; 
    long double (*f)(long double);
};

struct thread_data *thread_data_array;

// End of data structures

long double *samples;
long double *results;

long double map_intervals(long double x, long double *interval_from, long double *interval_to){
    x -= interval_from[0];
    x /= (interval_from[1] - interval_from[0]);
    x *= (interval_to[1] - interval_to[0]);
    x += interval_to[0];
    return x;
}

long double *uniform_sample(long double *interval, long double *samples, int size){
    for(int i = 0; i < size; i++){
        samples[i] = map_intervals((long double) rand(),
                                   rand_interval,
                                   interval);
    }
    return samples;
}

void print_array(long double *sample, int size){
    printf("array of size [%d]: [", size);

    for(int i = 0; i < size; i++){
        printf("%Lf", sample[i]);

        if(i != size - 1){
            printf(", ");
        }
    }

    printf("]\n");
}

long double monte_carlo_integrate(long double (*f)(long double), long double *samples, int size){
    // Your sequential code goes here
    long double accumulator = 0;
    //long double result;

    for (int i = 0; i < size; i++){
        accumulator += f(samples[i]);   
    }

    //result = accumulator/size;
    return accumulator/size;
}

void *monte_carlo_integrate_thread(void *args){
    // Your pthreads code goes here

    struct thread_data *my_data;
    int task_id, sample_begin, sample_end;
    long double (*function)(long double);
    long double accumulator = 0;    
    long double *t_sample, *t_result;


    my_data = (struct thread_data *) args;
    task_id = my_data->thread_id;
    function = my_data->f;
    sample_begin = my_data->sample_begin;
    sample_end = my_data->sample_end;
    t_sample = my_data->sample;
    t_result = my_data->result;

    for(int i = sample_begin; i < sample_end; i++)
        accumulator += function(t_sample[i]);
    

    t_result[task_id] = accumulator;

    pthread_exit((void *)task_id);
    //return 0;
}

int main(int argc, char **argv){
    if(argc != 4){
        printf(usage_message);
        exit(-1);
    } else if(atoi(argv[2]) >= FUNCTIONS || atoi(argv[2]) < 0){
        printf("Error: FUNCTION_ID must in [0,%d]\n", FUNCTIONS - 1);
        printf(usage_message);
        exit(-1);
    } else if(atoi(argv[3]) < 0){
        printf("Error: I need at least 1 thread\n");
        printf(usage_message);
        exit(-1);
    }

    if(DEBUG){
        printf("Running on: [debug mode]\n");
        printf("Samples: [%s]\n", argv[1]);
        printf("Function id: [%s]\n", argv[2]);
        printf("Threads: [%s]\n", argv[3]);
        printf("Array size on memory: [%.2LFGB]\n", ((long double) atoi(argv[1]) * sizeof(long double)) / 1000000000.0);
    }

    srand(time(NULL));

    int size = atoi(argv[1]);
    struct function target_function = functions[atoi(argv[2])];
    int n_threads = atoi(argv[3]);

    samples = malloc(size * sizeof(long double));

    long double estimate;

    if(n_threads == 1){
        if(DEBUG){
            printf("Running sequential version\n");
        }

        timer.c_start = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
        gettimeofday(&timer.v_start, NULL);

        estimate = monte_carlo_integrate(target_function.f,
                                         uniform_sample(target_function.interval,
                                                        samples,
                                                        size),
                                         size);

        timer.c_end = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
        gettimeofday(&timer.v_end, NULL);
    } else {
        if(DEBUG){
            printf("Running parallel version\n");
        }

        timer.c_start = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
        gettimeofday(&timer.v_start, NULL);

        // Your pthreads code goes here
        int error_code, t, i_start, i_end, step; 

        //memory allocation
        results = malloc((n_threads) * sizeof(long double));
        thread_data_array = malloc(n_threads * sizeof(struct thread_data));

        pthread_t threads[n_threads];
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        //sampling
        samples = uniform_sample(target_function.interval, samples, size);

        step = size/n_threads;
        i_start = 0;
        i_end = i_start + step;

        for(t = 0; t < n_threads; t++){
            thread_data_array[t].thread_id = t;
            thread_data_array[t].f = target_function.f;
            thread_data_array[t].sample_begin = i_start; //inclusive
            thread_data_array[t].sample_end = i_end; //exclusive
            thread_data_array[t].sample = samples;
            thread_data_array[t].result = results;

            error_code = pthread_create(&threads[t], &attr, 
                                        monte_carlo_integrate_thread, (void *) &thread_data_array[t]);

            if (error_code){
                printf("ERROR; return code from pthread_create() is %d\n", error_code);
                exit(-1);
            } 

            // update intervals
            i_start = i_end;
            i_end = (i_end + step >= size) ? size - 1 : i_end + step;  
        }

        pthread_attr_destroy(&attr);

        //join loop
        for(t = 0; t < n_threads; t++){
            error_code = pthread_join(threads[t], NULL);
            if(error_code){
                printf("ERROR; return code from pthread_join() is %d\n", error_code);
                exit(-1);
            };
        };

        //sum results
        estimate = 0.0;
        for(t = 0; t < n_threads; t++)
        	estimate += results[t];        
        estimate /= size;

        // Your pthreads code ends here

        timer.c_end = clock();
        clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
        gettimeofday(&timer.v_end, NULL);

        if(DEBUG && VERBOSE){
            print_array(results, n_threads);
        }
    }

    if(DEBUG){
        if(VERBOSE){
            print_array(samples, size);
            printf("Estimate: [%.33LF]\n", estimate);
        }
        printf("%.16LF, [%f, clock], [%f, clock_gettime], [%f, gettimeofday]\n",
               estimate,
               (double) (timer.c_end - timer.c_start) / (double) CLOCKS_PER_SEC,
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0,
               (double) (timer.v_end.tv_sec - timer.v_start.tv_sec) +
               (double) (timer.v_end.tv_usec - timer.v_start.tv_usec) / 1000000.0);
    } else {
        printf("%.16LF, %f\n",
               estimate,
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);
    }
    return 0;
}
