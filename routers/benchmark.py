from optimizers import scipy_bfgs
from controllers.benchmark import BenchmarkController


class BenchmarkRouter:
    @staticmethod
    def route(argv):
        benchmarks = []
        labels = []
        if (len(argv) == 2):
            benchmarks = [
                scipy_bfgs.ScipyBFGS.minimize,
            ]
            labels = [
                scipy_bfgs.ScipyBFGS.label,
            ]
        else:
            for i in range(3, len(argv)):
                if argv[i] == 'scipy-bfgs':
                    benchmarks.append(scipy_bfgs.ScipyBFGS.minimize)
                    labels.append(scipy_bfgs.ScipyBFGS.label)
                else:
                    raise AssertionError(
                        'Did not recognize argument "%s"' % argv[i])
        for optimizer, label in zip(benchmarks, labels):
            BenchmarkController.benchmark(optimizer, label)
