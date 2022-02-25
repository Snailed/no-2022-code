from optimizers import scipy_bfgs, scipy_nelder_mead, scipy_trust_ncg, line_search
from controllers.benchmark import BenchmarkController


class BenchmarkRouter:
    @staticmethod
    def route(argv):
        benchmarks = []
        labels = []
        if (len(argv) == 2):
            benchmarks = [
                scipy_bfgs.ScipyBFGS.minimize,
                scipy_nelder_mead.ScipyNelderMead.minimize,
                scipy_trust_ncg.ScipyTrustNCG.minimize,
                line_search.steepest_descent.SteepestDescentLineSearch.minimize,
                line_search.newton.NewtonLineSearch.minimize
            ]
            labels = [
                scipy_bfgs.ScipyBFGS.get_label(),
                scipy_nelder_mead.ScipyNelderMead.get_label(),
                scipy_trust_ncg.ScipyTrustNCG.get_label(),
                line_search.steepest_descent.SteepestDescentLineSearch.get_label(),
                line_search.newton.NewtonLineSearch.get_label()
            ]
        else:
            for i in range(2, len(argv)):
                if argv[i] == 'scipy-bfgs':
                    benchmarks.append(scipy_bfgs.ScipyBFGS.minimize)
                    labels.append(scipy_bfgs.ScipyBFGS.get_label())
                elif argv[i] == 'scipy-nelder-mead':
                    benchmarks.append(
                        scipy_nelder_mead.ScipyNelderMead.minimize)
                    labels.append(
                        scipy_nelder_mead.ScipyNelderMead.get_label())
                elif argv[i] == 'scipy-trust-ncg':
                    benchmarks.append(scipy_trust_ncg.ScipyTrustNCG.minimize)
                    labels.append(scipy_trust_ncg.ScipyTrustNCG.get_label())
                elif argv[i] == 'steepest-descent':
                    benchmarks.append(
                        line_search.steepest_descent.SteepestDescentLineSearch.minimize)
                    labels.append(
                        line_search.steepest_descent.SteepestDescentLineSearch.get_label())
                elif argv[i] == 'newton':
                    benchmarks.append(
                        line_search.newton.NewtonLineSearch.minimize)
                    labels.append(
                        line_search.newton.NewtonLineSearch.get_label())
                else:
                    raise AssertionError(
                        'Did not recognize argument "%s"' % argv[i])
        for optimizer, label in zip(benchmarks, labels):
            BenchmarkController.benchmark(optimizer, label)
