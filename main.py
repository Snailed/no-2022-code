import sys
import matplotlib.pyplot as plt
from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
from utils import validate_gradient


def main():
    plt.axes(projection='3d')
    plt.show()


def test():
    sum_of_different_powers.test()
    ellipsoid.test()
    rosenbrock.test()
    log_ellipsoid.test()
    attractive_sector.test()
    validate_gradient.test()


if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        main()
    elif sys.argv[1] == 'test':
        test()
    else:
        print('Usage: python main.py <ARGUMENT> \n Arguments:\n \t plot \t Plots the functions \n \t test \t Runs the tests of the implementation\n')
