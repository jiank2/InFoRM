import utils

from scipy.sparse import identity
from scipy.sparse.csgraph import laplacian


class DebiasResult:
    """
    debiasing the mining result
    """
    def __init__(self, tol=1e-12, maxiter=100):
        """
        initialize parameters for conjugate gradient
        :param tol: tolerance
        :param maxiter: maximum number of iterations
        """
        self.tol = tol
        self.maxiter = maxiter

    def matrix_conjugate_gradient(self, a, b, x):
        """
        solve linear system AX=B in matrix form
        :param a: matrix A
        :param b: matrix B
        :param x: matrix X
        :return: solution to the linear system
        """
        if self.maxiter is None:
            self.maxiter = a.shape[0] * 10

        r = b - a @ x
        p = r.copy()
        for _ in range(self.maxiter):
            residual_prev = utils.trace(r.T @ r)
            coeff = residual_prev / utils.trace(p.T @ a @ p)
            x += (coeff * p)
            r -= (coeff * a @ p)
            residual = utils.trace(r.T @ r)
            if residual < self.tol:
                return x
            beta = residual / residual_prev
            p = r + beta * p
        return x

    def fit(self, res, sim, alpha):
        """
        debias the mining results by solving (I + alpha * L_S) X = Y
        :param res: mining result Y
        :param sim: similarity matrix S
        :param lambda_: regularization parameter
        :return: debiased mining result X
        """
        if alpha == 0:
            return res
        lap = laplacian(sim)
        a = identity(sim.shape[0]) + alpha * lap
        b, x = res, res.copy()
        return self.matrix_conjugate_gradient(a, b, x)
