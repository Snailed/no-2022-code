from controllers import test, plot
from routers.plot import PlotRouter
from routers.test import TestRouter
from routers.benchmark import BenchmarkRouter
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        PlotRouter.route(sys.argv)
    elif sys.argv[1] == 'test':
        TestRouter.route(sys.argv)
    elif sys.argv[1] == 'benchmark':
        BenchmarkRouter.route(sys.argv)
    else:
        print('''Usage: python main.py <ARGUMENT> \n
              Arguments:\n
              \t plot <function> \t Plots the functions (ellipsoid, rosenbrock,
                        log-ellipsoid, attractive-sector, different-powers)\n
              \t test [tests...]\t Runs the tests of the implementation.\n
              \t benchmark [optimization-functions...]\t Benchmark one or more optimization functions and print output to stdout\n
              ''')
