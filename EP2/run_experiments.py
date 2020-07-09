import subprocess


def run_cuda(repetitions=15, dimensions=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)], size=4096):
    subprocess.run(['make', 'mandelbrot_cuda'])
    string = "dimensions,duration\n"
    for x, y in dimensions:
        for _ in range(repetitions):
            res = subprocess.run(['./mandelbrot_cuda',
                                  "-0.188", "-0.012", "0.554", "0.754",
                                  str(size), str(x), str(y)],
                                 capture_output=True)
            string += '"' + str(x) + ', '+str(y)+'", '
            string += str(float(res.stdout))+'\n'

    with open('cuda_experiments.csv', 'w+') as handle:
        handle.write(string)
        handle.close()


def run_ompi(repetitions=15, process_range=[2, 3, 5, 9, 17, 33, 65], size=4096):
    subprocess.run(['make', 'mandelbrot_ompi'])
    string = "processes,duration\n"
    for p in process_range:
        for _ in range(repetitions):
            res = subprocess.run([
                "mpirun", "-np", str(p), "--host", "localhost:" +
                str(p), "mandelbrot_ompi",
                "-0.188", "-0.012", "0.554", "0.754", str(size)],
                capture_output=True)

            string += str(p) + ','
            string += str(float(res.stdout))+'\n'

    with open('ompi_experiments.csv', 'w+') as handle:
        handle.write(string)
        handle.close()


def run_ompi_omp(repetitions=15, process_range=[2, 3, 5, 9, 17, 33, 65],
                 thread_range=[1, 2, 4, 8, 16, 32, 64], size=4096):
    subprocess.run(['make', 'mandelbrot_ompi_omp'])
    string = "processes,threads,duration\n"
    for p in process_range:
        for t in thread_range:
            for _ in range(repetitions):
                res = subprocess.run([
                    "mpirun", "-np", str(p), "--host", "localhost:" + str(p),
                    "mandelbrot_ompi_omp", "-0.188", "-0.012", "0.554",
                    "0.754", str(size), str(t)],
                    capture_output=True)

                string += str(p) + ',' + str(t) + ','
                string += str(float(res.stdout))+'\n'

    with open('ompi_omp_experiments.csv', 'w+') as handle:
        handle.write(string)
        handle.close()


if __name__ == "__main__":
    run_cuda()
    run_ompi()
    run_ompi_omp()
