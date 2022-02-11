import sys
import matplotlib.pyplot as plt
import numpy as np
from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
from utils import validate_gradient


def plot(arg):
    fs = [ellipsoid.Ellipsoid.f, rosenbrock.Rosenbrock.f, log_ellipsoid.LogEllipsoid.f, attractive_sector.AttractiveSector.f, sum_of_different_powers.SumOfDifferentPowers.f]
    
    if arg == 'ellipsoid':
        f = fs[0]
        label = 'f1(x)'
    elif arg == 'rosenbrock':
        f = fs[1]
        label = 'f2(x)'
    elif arg == 'log-ellipsoid':
        f = fs[2]
        label = 'f3(x)'
    elif arg == 'attractive-sector':
        f = fs[3]
        label = 'f4(x)'
    elif arg == 'different-powers':
        f = fs[4]
        label = 'f5(x)'
    else:
        print('Unknown function (ellipsoid, rosenbrock, log-ellipsoid, attractive-sector, different-powers)')
        return
    x, y = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            Z[i][j] = f([X[i][j], Y[i][j]])
    # for i, X_i in enumerate(X):
    #     for j, Y_j in enumerate(Y):
    #         z.append(f([X_i, Y_j]))
    # z = np.zeros((100, 100))
    # for i in range(100):
    #     for j in range(100):
    #         z[i][j] = f([x[i],y[i]])

    fig = plt.figure()
    plt.axes(projection='3d')
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, Z, 1000)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(label)
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
        if len(sys.argv) == 2:
            print()
        plot(sys.argv[2])
    elif sys.argv[1] == 'test':
        test()
    else:
        print('Usage: python main.py <ARGUMENT> \n Arguments:\n \t plot <function> \t Plots the functions (ellipsoid, rosenbrock, log-ellipsoid, attractive-sector, different-powers\n \t test \t Runs the tests of the implementation\n')
