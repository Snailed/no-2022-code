def gradient_convergence(x, f_grad, convergence=0.01):
    if f_grad(x) < 0.01:
        return True
    return False
