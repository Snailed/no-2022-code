import numpy as np


def trust_region(f, f_grad, f_hess, x_init, direction_f, delta_f, acceptance_f, callback=None, delta_init=1, max_iterations=10000):
    delta = delta_init
    x = x_init

    def m(p, x):
        return f(x) + np.array(f_grad(x)).T @ p + 0.5 * p.T @ f_hess(x) @ p
    for i in range(max_iterations):
        p, lam = direction_f(x, f, f_grad, f_hess, delta)
        print("-----")
        print("norm of p: ", np.linalg.norm(p))
        print("delta: ", delta)
        # print(p, lam)
        delta = delta_f(delta, m, f, x, p)
        if acceptance_f(f, m, x, p):
            x = x + p
        if callback and callback(x):
            break
    return x


def rho(f, m, x, p):
    # print("top part", f(x) - f(x+p))
    # print("bottom part", m(np.zeros(len(x)), x) - m(p, x))
    return (f(x) - f(x+p))/(m(np.zeros(len(x)), x) - m(p, x))


def adjust_trust_region(delta_init, rho, p_norm, delta_max=10000):
    delta = delta_init
    if rho < 0.25:
        delta = 0.25 * delta
    elif rho > 0.75 and np.around(p_norm, 5) == np.around(delta, 5):
        delta = min(2*delta, delta_max)
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
    # print(np.linalg.eigvals(X - np.diag(aug)))
    # print(np.linalg.cholesky(X - np.diag(aug)))
    return X - np.diag(aug)


# def trust_region_subproblem(lambda_init, delta, g, B, max_iterations=1000):
#     lam = lambda_init
#     iters = 0
#     for l in range(max_iterations):
#         iters = 0
#         smallest_eigenval = np.min(np.linalg.eigvals(B))
#         while lam <= -smallest_eigenval:
#             lam = -smallest_eigenval + 0.01
#             iters += 1
#             if iters > 100:
#                 print("break")
#                 # break
#         assert lam > - smallest_eigenval
#         R = np.linalg.cholesky(
#             B + np.diag(np.array([lam for _ in g])))
#         p = np.linalg.solve(R.T @ R, -g)
#         q = np.linalg.solve(R.T, p)
#         lam = lam + (np.linalg.norm(p)/np.linalg.norm(q))**2 * \
#             ((np.linalg.norm(p) - delta) / delta)
#         iters = max(iters, l)
#     return p, lam

def trust_region_subproblem(lambda_init, delta, g, B, max_iterations=10):
    lam = lambda_init
    iters = 0
    if np.min(np.linalg.eigvals(B)) > 0:
        lam = 0
        if np.linalg.norm(-np.linalg.inv(B) @ g) <= delta:
            return -np.linalg.inv(B) @ g, lam
        else:
            print("The newton step is outside the trust region",
                  np.linalg.norm(-np.linalg.inv(B) @ g))
    else:
        print("the smallest eigenvalue negative or 0")
    for l in range(max_iterations):
        smallest_eigenval = np.min(np.linalg.eigvals(B))
        if smallest_eigenval <= 0:
            lam = -smallest_eigenval + 0.001
        eigenvalues, Q = np.linalg.eig(B + lam * np.eye(B.shape[0]))

        def norm_p_squared(Q, g, eigenvalues, lam):
            return np.sum([((Q[:, i].T @ g)/(eigenvalues[i] + lam))**2 for i in range(len(eigenvalues))])
        phi = norm_p_squared(Q, g, eigenvalues, lam) - delta
        phi_prime = -2 * np.sum([((Q[:, i].T @ g)**2/(eigenvalues[i] + lam)**3)
                                for i in range(len(eigenvalues))])
        lam = lam - phi/phi_prime
        iters += 1
        p = -np.add.reduce([((Q[:, i].T @ g)/(eigenvalues[i] + lam)) * Q[:, i]
                            for i in range(len(eigenvalues))])
        print("p", p)
        print("||p||", np.linalg.norm(p))
    print("----")
    return p, lam

# def trust_region_subproblem(lambda_init, delta, g, B, max_iterations=5):
#     print("DELTA", delta)
#     lam = lambda_init
#     iters = 0
#     for l in range(max_iterations):
#         smallest_eigenval = np.min(np.linalg.eigvals(B))
#         if smallest_eigenval <= 0:
#             lam = -smallest_eigenval + 0.001
#         eigenvalues, Q = np.linalg.eig(B + lam * np.eye(B.shape[0]))
#         assert np.all(np.around((B + lam * np.eye(B.shape[0])
#                                  ) @ Q[:, 0], 5) == np.around(eigenvalues[0] * Q[:, 0], 5)), "First eigenvector/value pair was not valid (Q = %s, eigenvalues = %s, LHS = %s, RHS = %s)" % (Q, eigenvalues, (B + lam * np.eye(B.shape[0])) @ Q[:, 1], eigenvalues[1] * Q[:, 1])

#         smallest_eigenval = np.min(
#             np.linalg.eigvals(B + lam * np.eye(B.shape[0])))
#         lam0 = -smallest_eigenval + 0.0001

#         def get_p(B, lam, g):
#             return -np.linalg.inv(B + lam * np.eye(B.shape[0])) @ g

#         n = 0
#         while np.linalg.norm(get_p(B, lam0, g)) <= delta:
#             lam0 /= 10
#             n += 1
#             assert n < 10000
#         assert np.linalg.norm(get_p(B, lam0, g)) > delta

#         lam1 = -smallest_eigenval + 1000000

#         while np.linalg.norm(get_p(B, lam1, g)) >= delta:
#             lam1 *= 10
#             n += 1
#             assert n < 10000
#         assert np.linalg.norm(get_p(B, lam1, g)) < delta

#         for i in range(10000):
#             lam = (lam0 + lam1)/2
#             if np.linalg.norm(get_p(B, lam, g)) > delta:
#                 lam0 = lam
#             else:
#                 lam1 = lam
#         p = get_p(B, lam, g)
#     return p, lam


def acceptance_criteria(f, m, x, p, eta=0.1):
    return rho(f, m, x, p) > eta


def test():
    from functions.log_ellipsoid import LogEllipsoid
    from functions.rosenbrock import Rosenbrock
    x = [2, 1]
    # delta = 10
    # lambda_init = 1
    # for function in [LogEllipsoid, Rosenbrock]:
    #     for delta in np.linspace(0.001, 1, 100):
    #         for lambda_init in np.linspace(0.00001, 1, 100):
    #             p, lam = trust_region_subproblem(
    #                 lambda_init, delta, function.gradient(x), function.hessian(x))
    #             lhs = (function.hessian(x) +
    #                    np.diag([lam for _ in range(len(x))])) @ p
    #             rhs = -1 * function.gradient(x)
    #             assert len(lhs) == len(rhs)
    #             for l, r in zip(lhs, rhs):
    #                 if np.around(l, 2) != np.around(r, 2):
    #                     print(l, r)
    #                 assert np.around(l, 2) == np.around(r, 2)
    #             if np.around(lam * (delta - np.linalg.norm(p)), 2) != 0:
    #                 print(function, delta, lambda_init)
    #                 print(lam * (delta - np.linalg.norm(p)))
    #             assert np.around(lam * (delta - np.linalg.norm(p)), 2) == 0
    #             assert np.all(np.linalg.eigvals(
    #                 function.hessian(x) + lam * np.eye(len(x))) >= 0)

    # test the subproblem
    def f(x):
        X = np.array(x)
        return np.sum(0.5 * X.T @ X)

    def f_grad(x):
        return np.array(x)

    def f_hess(x):
        return np.eye(len(x))

    init_x = [1, 1]
    delta = 100
    p, lam = trust_region_subproblem(1, delta, f_grad(init_x), f_hess(init_x))
    # assert delta - np.linalg.norm(p) == 0
    assert lam == 0, ("Lambda is not zero - it is %f" % lam)

    init_x = [1, 1]
    delta = 0.5
    p, lam = trust_region_subproblem(1, delta, f_grad(init_x), f_hess(init_x))
    assert delta - \
        np.linalg.norm(
            p) == 0, "The norm of p is not the same as delta, ||p|| = %f, delta = %d" % (np.linalg.norm(p), delta)
