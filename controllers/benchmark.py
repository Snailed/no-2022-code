from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
import numpy as np


def benchmark_ellipsoid(minimize, x0, alpha=10000, callback=None):
    def arg1(x):
        return ellipsoid.Ellipsoid.f(x, alpha=alpha)

    def arg2(x):
        return ellipsoid.Ellipsoid.gradient(x, alpha=alpha)
    return minimize(arg1, x0, arg2, callback=callback)


def benchmark_rosenbrock(minimize, x0, callback=None):
    return minimize(
        lambda x: rosenbrock.Rosenbrock.f(
            x), x0, rosenbrock.Rosenbrock.gradient, callback=callback
    )


def benchmark_log_ellipsoid(minimize, x0, epsilon=0.0001, alpha=10000, callback=None):
    return minimize(
        lambda x: log_ellipsoid.LogEllipsoid.f(
            x, alpha=alpha, epsilon=epsilon),
        x0,
        lambda x: log_ellipsoid.LogEllipsoid.gradient(
            x, alpha=alpha, epsilon=epsilon),
        callback=callback
    )


def benchmark_attractive_sector(minimize, x0, q=10000, callback=None):
    return minimize(
        lambda x: attractive_sector.AttractiveSector.f(
            x, q=q
        ),
        x0,
        lambda x: attractive_sector.AttractiveSector.gradient(
            x, q=q
        ),
        callback=callback
    )


def benchmark_sum_of_different_powers(minimize, x0, callback=None):
    return minimize(
        lambda x: sum_of_different_powers.SumOfDifferentPowers.f(
            x
        ),
        x0,
        lambda x: sum_of_different_powers.SumOfDifferentPowers.gradient(
            x
        ),
        callback=callback
    )


class BenchmarkController:
    @staticmethod
    def benchmark(minimize, label):
        # Evaluate on all functions
        # Metrics:
        # 1: How far does the results deviate from a known solution? What value does the gradient have at a returned solution?
        # 2: How does the optimizer perform on different hyperparameters? Does it always give the correct answer for all functions in its class? How about in multiple dimensions? (robustness)
        # 3: How does the optimizer perform on different starting points?
        # 4: How many iterations does it take before the algorithm converges? (Efficiency)
        ds = [2, 3, 4, 10, 100]
        n = 100
        for d in ds:
            iterations = []  # iterations needed to return result
            gradients = []  # gradient at returned point
            x0s = []
            x0s.append(np.zeros(d))
            for i in range(0, n):
                # Add random points with each coordinate in [-2, 2]
                x0s.append((np.random.rand(d) - 0.5)*4)
            for x0 in x0s:
                print('benchmark ellipsoid')
                _, i, grad = benchmark_ellipsoid(minimize, x0, alpha=10000)
                iterations.append(i)
                # if np.linalg.norm(grad) > 1:
                #     print('Ellipsoid')
                #     print(x0)
                print('benchmark log ellipsoid')
                gradients.append(np.linalg.norm(grad))
                _, i, grad = benchmark_log_ellipsoid(
                    minimize, x0*0.0001, alpha=10000, epsilon=0.01
                )
                iterations.append(i)
                # if np.linalg.norm(grad) > 1:
                #     print('Log Ellipsoid')
                #     print(np.linalg.norm(grad))
                #     print(i)
                print('benchmark rosenbrock')
                gradients.append(np.linalg.norm(grad))
                if d == 2:  # Rosenbrock is only supported in 2D
                    _, i, grad = benchmark_rosenbrock(
                        minimize, x0)
                    iterations.append(i)
                    # if np.linalg.norm(grad) > 1:
                    #     print('Rosenbrock')
                    #     print(x0)
                    gradients.append(np.linalg.norm(grad))
                if d == 1:  # Attractive Sector is only supported in 1D
                    print('benchmark attractive sector')
                    _, i, grad = benchmark_attractive_sector(
                        minimize, x0, q=10
                    )
                    iterations.append(i)
                # if np.linalg.norm(grad) > 1:
                #     print('Attractive sector')
                #     print(x0)
                #     print(grad)
                #     print(i)
                print('benchmark sum of different powers')
                gradients.append(np.linalg.norm(grad))
                _, i, grad = benchmark_sum_of_different_powers(
                    # move the points such that we don't get complex numbers
                    minimize, x0
                )
                iterations.append(i)
                # if np.linalg.norm(grad) > 1:
                #     print('Sum of different powers')
                #     print(x0)
                gradients.append(np.linalg.norm(grad))
            print('Optimization Algorithm Benchmark for "%s"' % label)
            print('PARAMETERS:')
            print('\tDimension: %s' % str(d))
            print('\tAmount of points used: %d' % n)
            print('\t...picked at random from the interval [-2, 2]')
            print('RESULTS:')
            print('\tMean iterations needed to return result %f' %
                  np.mean(iterations))
            print('\tVariance of iterations needed to return result %f' %
                  np.var(iterations))
            print('\tMean gradient norm at returned result %f' %
                  np.mean(gradients))
            print('\tVariance of gradient norm at returned result %f' %
                  np.var(gradients))
