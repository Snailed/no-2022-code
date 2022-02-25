from controllers.test import TestController
from functions import ellipsoid, rosenbrock, log_ellipsoid, attractive_sector, sum_of_different_powers
from utils import validate_gradient
from optimizers.line_search import direction, line_search, steepest_descent, step_size, newton


class TestRouter:
    @staticmethod
    def route(argv):
        if (len(argv) == 2):
            return TestController.test([
                sum_of_different_powers.test,
                ellipsoid.test,
                rosenbrock.test,
                log_ellipsoid.test,
                attractive_sector.test,
                validate_gradient.test,
                step_size.backtracking.test,
                direction.test,
                steepest_descent.test,
                newton.test
            ])
        else:
            tests = []
            for i in range(2, len(argv)):
                if argv[i] == 'ellipsoid':
                    tests.append(ellipsoid.test)
                elif argv[i] == 'rosenbrock':
                    tests.append(rosenbrock.test)
                elif argv[i] == 'log-ellipsoid':
                    tests.append(log_ellipsoid.test)
                elif argv[i] == 'attractive-sector':
                    tests.append(attractive_sector.test)
                elif argv[i] == 'different-powers':
                    tests.append(sum_of_different_powers.test)
                elif argv[i] == 'gradient':
                    tests.append(validate_gradient.test)
                elif argv[i] == 'backtrack':
                    tests.append(step_size.backtracking.test)
                elif argv[i] == 'line-search-direction':
                    tests.append(direction.test)
                elif argv[i] == 'line-search':
                    tests.append(line_search.test)
                elif argv[i] == 'steepest-descent':
                    tests.append(steepest_descent.test)
                elif argv[i] == 'newton':
                    tests.append(newton.test)
                else:
                    raise AssertionError(
                        'Did not recognize argument %s' % argv[i])
            return TestController.test(tests)
