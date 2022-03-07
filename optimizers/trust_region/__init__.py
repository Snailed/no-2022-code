import numpy as np


def trust_region(f, f_grad, f_hess, x_init, direction_f, delta_f, acceptance_f, callback=None, delta_init=1, max_iterations=10000):
    delta = delta_init
    x = x_init

    def m(p, x):
        return f(x) + p.T @ f_grad(x) + p.T @ f_hess(x) @ p
    for i in range(max_iterations):
        p, lam = direction_f(x, f, f_grad, f_hess, delta)
        delta = delta_f(delta, m, f, x, p)
        if acceptance_f(f, m, x, p):
            x = x + p
        if callback and callback(x):
            break
    return x


def rho(f, m, x, p):
    return (f(x) - f(x+p))/(m(np.zeros(len(x)), x) - m(p, x))


def adjust_trust_region(delta_init, rho, p_norm, delta_max=10000):
    delta = delta_init
    if rho < 0.25:
        delta = 0.25 * delta_init
    elif rho > 0.75 and p_norm == delta_init:
        delta = min(2*delta_init, delta_max)
    return delta


def check_pd_or_augment(X):
    eigvals = np.linalg.eigvals(X)
    smallest_eigenvalue = min(eigvals)
    aug = np.zeros(len(eigvals))
    if smallest_eigenvalue < 0:
        aug = np.full(len(eigvals), smallest_eigenvalue - 0.1)
    # for i, e in enumerate(eigvals):
    #     if e < 0:
    #         aug[i] = e - 1
    print(np.linalg.eigvals(X - np.diag(aug)))
    print(np.linalg.cholesky(X - np.diag(aug)))
    return X - np.diag(aug)


def trust_region_subproblem(lambda_init, delta, g, B, max_iterations=100):
    lam = lambda_init
    for l in range(max_iterations):
        # inv = np.linalg.inv(
        #     B + np.diag(np.array([lam for _ in g]))
        # )
        # p = (
        #     inv @ (-1 * g)
        # )
        # print("LAMDA: ", lam, "B: ", B)
        # print("iteration", l)
        iters = 0
        smallest_eigenval = np.min(np.linalg.eigvals(B))
        while lam < - smallest_eigenval:
            lam = lam - (lam + smallest_eigenval)
            iters += 1
            if iters > 100:
                print("break")
                break
        assert lam >= - np.min(np.linalg.eigvals(B))

        # print("B", B)
        # print("smallest_eigenval", smallest_eigenval)
        # print("lambda", lam)
        # print("augmented B", B + np.diag(np.array([lam + 0.0001 for _ in g])))
        # print("augmented B eigenvalues", np.linalg.eigvals(B + np.diag(np.array([lam + 0.0001 for _ in g]))))
        R = np.linalg.cholesky(
            B + np.diag(np.array([lam + 0.0001 for _ in g])))
        p = np.linalg.solve(R.T @ R, -g)
        q = np.linalg.solve(R.T, p)

#         q = np.linalg.inv(R) @ p
        # if not (np.all(np.round(R.T @ q, 3) == np.round(p, 3))):
        #     print("R", R, "q", q)
        #     print("R.T @ q", R.T @ q, "p", p)
        #     print("around R.T @ q", np.round(R.T @ q, 3), "p", np.round(p, 3))
        #     print("equality R.T @ q", np.round(R.T @ q, 3) == np.round(p, 3))
        #     assert np.all(np.round(R.T @ q, 3) == np.round(p, 3))
        # print("p: ", p, "q: ", q, "q@-g", q @ -g)
        lam = lam + (np.linalg.norm(p)/np.linalg.norm(q))**2 * \
            ((np.linalg.norm(p) - delta) / delta)
    return p, lam


def acceptance_criteria(f, m, x, p, eta=0.2):
    return rho(f, m, x, p) > eta


def test():
    from functions.log_ellipsoid import LogEllipsoid
    from functions.rosenbrock import Rosenbrock
    x = [2, 1]
    # delta = 10
    # lambda_init = 1
    for function in [LogEllipsoid, Rosenbrock]:
        for delta in np.linspace(0.001, 1, 100):
            for lambda_init in np.linspace(0.00001, 1, 100):
                p, lam = trust_region_subproblem(
                    lambda_init, delta, function.gradient(x), function.hessian(x))
                lhs = (function.hessian(x) +
                       np.diag([lam for _ in range(len(x))])) @ p
                rhs = -1 * function.gradient(x)
                assert len(lhs) == len(rhs)
                for l, r in zip(lhs, rhs):
                    if np.around(l, 2) != np.around(r, 2):
                        print(l, r)
                    assert np.around(l, 2) == np.around(r, 2)
                if np.around(lam * (delta - np.linalg.norm(p)), 2) != 0:
                    print(function, delta, lambda_init)
                    print(lam * (delta - np.linalg.norm(p)))
                assert np.around(lam * (delta - np.linalg.norm(p)), 2) == 0
                assert np.all(np.linalg.eigvals(
                    function.hessian(x) + lam * np.eye(len(x))) >= 0)
