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


if __name__ == "__main__":
    run_cuda(3, [(2, 2), (4, 4)], 256)
