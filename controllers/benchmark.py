from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
import numpy as np


def benchmark_ellipsoid(minimize, x0, alpha=10000):
    return minimize(
        lambda x: ellipsoid.Ellipsoid.f(x, alpha), x0, lambda x: ellipsoid.Ellipsoid.gradient(x, alpha))


def benchmark_rosenbrock(minimize, x0):
    return minimize(
        lambda x: rosenbrock.Rosenbrock.f(
            x), x0, rosenbrock.Rosenbrock.gradient
    )


def benchmark_log_ellipsoid(minimize, x0, epsilon=0.0001, alpha=10000):
    return minimize(
        lambda x: log_ellipsoid.LogEllipsoid.f(
            x, alpha=alpha, epsilon=epsilon),
        x0,
        lambda x: log_ellipsoid.LogEllipsoid.gradient(
            x, alpha=alpha, epsilon=epsilon)
    )


def benchmark_attractive_sector(minimize, x0, q=10000):
    return minimize(
        lambda x: attractive_sector.AttractiveSector.f(
            x, q=q
        ),
        x0,
        lambda x: attractive_sector.AttractiveSector.gradient(
            x, q=q
        )
    )


def benchmark_sum_of_different_powers(minimize, x0):
    return minimize(
        lambda x: sum_of_different_powers.SumOfDifferentPowers.f(
            x
        ),
        x0,
        lambda x: sum_of_different_powers.SumOfDifferentPowers.gradient(
            x
        )
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
        iterations = []  # iterations needed to return result
        gradients = []  # gradient at returned point
        for d in ds:
            x0s = []
            x0s.append(np.zeros(d))
            for i in range(0, n):
                # Add random points with each coordinate in [-2, 2]
                x0s.append((np.random.rand(d) - 0.5)*4)
            for x0 in x0s:
                if 0.0 not in x0:
                    _, i, grad = benchmark_ellipsoid(minimize, x0, alpha=10000)
                    iterations.append(i)
                    gradients.append(i)
                    _, i, grad = benchmark_log_ellipsoid(
                        minimize, x0, alpha=10000, epsilon=0.0001
                    )
                    iterations.append(i)
                    gradients.append(i)
                if d == 2:  # Rosenbrock is only supported in 2D
                    _, i, grad = benchmark_rosenbrock(
                        minimize, x0)
                    iterations.append(i)
                    gradients.append(i)
                _, i, grad = benchmark_attractive_sector(
                    minimize, x0, q=10000
                )
                iterations.append(i)
                gradients.append(i)
                _, i, grad = benchmark_sum_of_different_powers(
                    minimize, x0
                )
                iterations.append(i)
                gradients.append(i)

        print('Optimization Algorithm Benchmark for "%s"' % label)
        print('PARAMETERS:')
        print('\tDimensions used: %s' % str(ds))
        print('\tAmount of points used: %d' % n)
        print('\t...picked at random from the interval [-2, 2]')
        print('RESULTS:')
        print('\tMean iterations needed to return result' %
              np.mean(iterations))
        print('\tVariance of iterations needed to return result' %
              np.mean(iterations))
        print('\tMean gradient at returned result' % np.mean(gradients))
        print('\tVariance of gradient at returned result' % np.var(gradients))
