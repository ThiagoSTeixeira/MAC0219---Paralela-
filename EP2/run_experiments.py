import subprocess


def run_seq(repetitions=15, size=4096):
    subprocess.run(["make", "mandelbrot_seq"])
    string = "duration\n"
    print("[sequential]: starting experiments")
    for _ in range(repetitions):
        res = subprocess.run(
            ["./mandelbrot_seq", "-0.188", "-0.012", "0.554", "0.754", str(size)],
            capture_output=True,
        )
        string += str(float(res.stdout)) + "\n"

    with open("seq_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()
    print("[sequential]: ending experiments")


def run_omp(repetitions=15, thread_range=[1, 2, 4, 8, 16, 32, 64], size=4096):
    subprocess.run(["make", "mandelbrot_omp"])
    string = "threads,duration\n"
    print("[omp]: starting experiments")
    for t in thread_range:
        print("[omp]:", t, "threads")
        for _ in range(repetitions):
            res = subprocess.run(
                [
                    "./mandelbrot_omp",
                    "-0.188",
                    "-0.012",
                    "0.554",
                    "0.754",
                    str(size),
                    str(t),
                ],
                capture_output=True,
            )
            string += str(t) + ","
            string += str(float(res.stdout)) + "\n"

    with open("omp_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()

    print("[omp]: ending experiments")


def run_pth(repetitions=15, thread_range=[1, 2, 4, 8, 16, 32, 64], size=4096):
    subprocess.run(["make", "mandelbrot_pth"])
    string = "threads,duration\n"
    print("[pthreads]: starting experiments")
    for t in thread_range:
        print("[pthreads]:", t, "threads")
        for _ in range(repetitions):
            res = subprocess.run(
                [
                    "./mandelbrot_pth",
                    "-0.188",
                    "-0.012",
                    "0.554",
                    "0.754",
                    str(size),
                    str(t),
                ],
                capture_output=True,
            )
            string += str(t) + ","
            string += str(float(res.stdout)) + "\n"

    with open("pth_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()

    print("[pthreads]: ending experiments")


def run_cuda(
    repetitions=15, dimensions=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)], size=4096
):
    subprocess.run(["make", "mandelbrot_cuda"])
    string = "dimensions,duration\n"
    for x, y in dimensions:
        print("[cuda]: running for: ", x, y)
        for _ in range(repetitions):
            res = subprocess.run(
                [
                    "./mandelbrot_cuda",
                    "-0.188",
                    "-0.012",
                    "0.554",
                    "0.754",
                    str(size),
                    str(x),
                    str(y),
                ],
                capture_output=True,
            )
            string += '"' + str(x) + ", " + str(y) + '", '
            string += str(float(res.stdout)) + "\n"

    with open("cuda_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()


def run_ompi(repetitions=15, process_range=[1, 2, 4, 8, 16, 32, 64], size=4096):
    subprocess.run(["make", "mandelbrot_ompi"])
    string = "processes,duration\n"
    for p in process_range:
        print("[ompi]: running for: ", p)
        for _ in range(repetitions):
            res = subprocess.run(
                [
                    "mpirun",
                    "-np",
                    str(p),
                    "--host",
                    "localhost:" + str(p),
                    "mandelbrot_ompi",
                    "-0.188",
                    "-0.012",
                    "0.554",
                    "0.754",
                    str(size),
                ],
                capture_output=True,
            )

            string += str(p) + ","
            string += str(float(res.stdout)) + "\n"

    with open("ompi_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()


def run_ompi_omp(
    repetitions=15,
    process_range=[1, 2, 4, 8, 16, 32, 64],
    thread_range=[1, 2, 4, 8, 16, 32, 64],
    size=4096,
):
    subprocess.run(["make", "mandelbrot_ompi_omp"])
    string = "processes,threads,duration\n"
    for p in process_range:
        for t in thread_range:
            print("[ompi_omp]: running for: ", p, t)
            for _ in range(repetitions):
                res = subprocess.run(
                    [
                        "mpirun",
                        "-np",
                        str(p),
                        "--host",
                        "localhost:" + str(p),
                        "mandelbrot_ompi_omp",
                        "-0.188",
                        "-0.012",
                        "0.554",
                        "0.754",
                        str(size),
                        str(t),
                    ],
                    capture_output=True,
                )

                string += str(p) + "," + str(t) + ","
                string += str(float(res.stdout)) + "\n"

    with open("ompi_omp_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()


def run_ompi_cuda(
    repetitions=15,
    process_range=[1, 2, 4, 8, 16, 32, 64],
    dimensions=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
    size=4096,
):
    subprocess.run(["make", "mandelbrot_ompi_cuda"])
    string = "processes,dimensions,duration\n"
    for p in process_range:
        for x, y in dimensions:
            print("[ompi_cuda]: running for: ", p, x, y)
            for _ in range(repetitions):
                res = subprocess.run(
                    [
                        "mpirun",
                        "-np",
                        str(p),
                        "--host",
                        "localhost:" + str(p),
                        "mandelbrot_ompi_cuda",
                        "-0.188",
                        "-0.012",
                        "0.554",
                        "0.754",
                        str(size),
                        str(x),
                        str(y),
                    ],
                    capture_output=True,
                )

                string += str(p) + ',"' + str(x) + ", " + str(y) + '",'
                string += str(float(res.stdout)) + "\n"

    with open("ompi_cuda_experiments.csv", "w+") as handle:
        handle.write(string)
        handle.close()


if __name__ == "__main__":
    run_seq()
    run_pth()
    run_omp()
    run_cuda()
    run_ompi()
    run_ompi_omp()
    run_ompi_cuda()
