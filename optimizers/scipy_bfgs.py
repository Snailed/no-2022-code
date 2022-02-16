import numpy as np
from scipy.optimize import minimize


class ScipyBFGS:
    @property
    def label(self):
        return 'SciPy BFGS'

    @staticmethod
    def minimize(f, x0, der):
        optimization_result = minimize(f, x0, method='BFGS', jac=der, tol=None, options={
            'gtol': 1e-2, 'disp': False})
        if optimization_result.success:
            return optimization_result.x, optimization_result.nit, optimization_result.jac
        else:
            print('Optimization failure message:')
            print(optimization_result.message)
            print('Optimization point:')
            print(optimization_result.x)
            print('Optimization starting point:')
            print(x0)
            raise AssertionError('Optimization failed!')
