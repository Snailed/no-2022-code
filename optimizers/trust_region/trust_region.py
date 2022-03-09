from optimizers.trust_region import trust_region, trust_region_subproblem, adjust_trust_region, rho, acceptance_criteria
import numpy as np


class TrustRegion:
    @staticmethod
    def get_label():
        return 'Newton Line Search'

    @staticmethod
    def minimize(
            f,
            x0,
            der,
            hes=None,
            max_iterations=100,
            callback=None,
            delta_init=100,
            lambda_init=5,
            meta_callback=None
    ):
        assert hes is not None
        meta = {}

        def default_callback(xk):
            if meta_callback:
                meta_callback(meta)
            if callback:
                return callback(xk)
            else:
                if np.linalg.norm(der(xk)) < 0.00001:
                    return True
                return False

        def direction_f(x, f, gradient, hessian, delta):
            meta['direction'] = trust_region_subproblem(
                lambda_init, delta, np.array(gradient(x)), hessian(x))
            return trust_region_subproblem(lambda_init, delta, np.array(gradient(x)), hessian(x))

        def delta_f(delta, m, f, x, p):
            r = rho(f, m, x, p)
            meta['rho'] = r
            meta['delta'] = adjust_trust_region(
                delta, r, np.linalg.norm(p), delta_max=10000)
            return meta['delta']

        return trust_region(f, der, hes, x0, direction_f, delta_f, acceptance_criteria, callback=default_callback, max_iterations=max_iterations, delta_init=delta_init)
