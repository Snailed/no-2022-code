from scipy.optimize import minimize


class ScipyNelderMead:
    @staticmethod
    def get_label():
        return 'SciPy Nelder Mead'

    @staticmethod
    def minimize(f, x0, der, callback=None):
        optimization_result = minimize(
            f, x0, method='Nelder-Mead', jac=der, tol=None, callback=callback, options={'disp': False})
        if optimization_result.success:
            return optimization_result.x, optimization_result.nit, der(optimization_result.x)
        else:
            if optimization_result.message == 'Warning: Desired error not necessarily achieved due to precision loss.' or optimization_result.message == 'Warning: Maximum number of iterations has been exceeded.' or optimization_result.message == 'Warning: CG iterations didn\'t converge. The Hessian is not positive definite.' or optimization_result.message == 'Maximum number of function evaluations has been exceeded.':
                return optimization_result.x, optimization_result.nit, der(optimization_result.x)
            else:
                print('Optimization failure message:')
                print(optimization_result.message)
                print('Optimization point:')
                print(optimization_result.x)
                print('Optimization starting point:')
                print(x0)
                raise AssertionError('Optimization failed!')
