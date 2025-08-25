import math
from numba import njit
import numpy as np
from scipy.special import gamma, hyp2f1
from tqdm import tqdm

from utils import g, b, cov, rmright1, ml, zeta

class gBergomi:
    """
    Class for generating paths of the gBergomi model.
    See https://github.com/ryanmccrickerd/rough_bergomi for the original implementation.
    """

    def __init__(self, n=100, N=1000, T=1.00, a=-0.4, b=1):
        """
        n: number of grid points per year
        N: number of paths
        T : number of years
        a: Hurst Parameter - 1/2
        b: beta from a one-sided M-Wright distribution
        """

        self.T = T
        self.n = n
        self.dt = 1.0 / self.n  # Step size
        self.s = int(self.n * self.T)  # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :]  # Time grid
        self.a = a
        self.b = b
        self.N = N

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, 1 + self.s))  # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s))  # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:, i] = dW[:, i - 1, 1]  # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s)  # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]  # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])

        # Extract appropriate part of convolution
        Y2 = GX[:, :1 + self.s]

        # Finally contruct and return full process
        Y = (Y1 + Y2)
        return Y

    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0):
        """
        Constructs correlated price Brownian increments.
        """
        self.rho = rho
        dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho ** 2) * dW2
        return dB

    def ml(self, x):
        '''
        Vectorised Mittag-Leffler function.
        '''
        n = np.arange(50)
        x = np.atleast_1d(x)  # Ensure t is treated as at least 1D
        terms = x[:, None] ** n / gamma(self.b * n + 1)  # Expand t for broadcasting
        return np.sum(terms, axis=-1)

    def V(self, Y, xi=1.0, eta=1.0):
        """
        gBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        c = 1 / gamma(a + 1)
        ybeta = rmright1(1, 1)
        V = xi * self.ml(eta ** 2 * c ** 2 * t[0] ** (2 * a + 1) / (4 * a + 2)) ** -1 * np.exp(
            eta * c * ybeta ** 0.5 * Y)
        return V

    def S(self, V, dB, S0=1):
        """
        gBergomi price process.
        """
        self.S0 = S0
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def S1(self, V, dW1, rho, S0=1):
        """
        gBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho ** 2 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

class gBergomiVIX:
    def __init__(self, n, N, H, eta, beta, xi, mc_sims, T, cutoff=8):
        """
        n: number of grid points per year
        N: number of grid points for approximation
        alpha: 2*Hurst parameter
        eta: vol-of-vol
        beta:  M-Wright beta
        mc_sims: number of MC simulations
        T : number of years
        cutoff: size of correlation matrix
        """
        self.n = n
        self.N = N
        self.H = H
        self.eta = eta
        self.beta = beta
        self.xi = xi
        self.mc_sims = mc_sims
        self.T = T
        self.cutoff = cutoff
        self.Delta = 1. / 12.

        self.n_T = int(np.floor(self.n * self.T)) # Number of grid points per year for VIX simulation
        self.time_grid = np.linspace(0, self.T, self.n_T, endpoint=True)  # Time points for VIX simulation

        self.H_minus = self.H - 0.5
        self.H_plus = self.H + 0.5
        self.const_c = 1. / math.gamma(self.H_plus)  # fBm constant
        self.const_b = self.eta ** 2 * self.const_c ** 2 / (4 * self.H)
        self.const = self.eta * self.const_c

        self.v_cov_matrix = np.zeros((self.n_T, cutoff, cutoff))  # Covariance matrix for Volterra process
        self.v_cov_matrix_chol = np.zeros((self.n_T, cutoff, cutoff))  # Cholesky decomposition for covariance matrix
        self.v_process = np.zeros((self.mc_sims, self.N))  # Volterra process

        self.frak_t = np.zeros((self.n_T, self.N))
        self.diff = np.zeros(self.n_T)

        self.corr = np.zeros((self.n_T, self.N - self.cutoff))
        self.var = np.zeros((self.n_T, self.N - self.cutoff))
        self.vix_const = np.zeros((self.n_T, self.N))

        for i in range(1, self.n_T):
            self.frak_t[i, :] = np.linspace(self.time_grid[i], self.time_grid[i] + self.Delta, self.N, endpoint=True)  # Time points for approximation
            self.diff[i] = float(self.frak_t[i, 1] - self.frak_t[i, 0])

            # Computing covariance matrix
            for j in range(self.cutoff):
                self.v_cov_matrix[i, j, j] = self._var_struct(self.frak_t[i][j], self.time_grid[i])
                for k in range(j + 1, self.cutoff):
                    self.v_cov_matrix[i, j, k] = self._cov_struct(self.frak_t[i][j], self.frak_t[i][k],
                                                                  self.time_grid[i])
                    self.v_cov_matrix[i, k, j] = self.v_cov_matrix[i, j, k]

            # Cholesky decomposition
            self.v_cov_matrix_chol[i] = np.linalg.cholesky(self.v_cov_matrix[i])

            # Precomputable terms in Volterra process simulation
            for j in range(self.cutoff, self.N):
                self.corr[i, j - self.cutoff] = self._corr_struct(self.frak_t[i][j - 1], self.frak_t[i][j],
                                                                  self.time_grid[i])
                self.var[i, j - self.cutoff] = self._var_struct(self.frak_t[i][j], self.time_grid[i]) ** 0.5

            # Precomputable terms in VIX calculation
            for j in range(self.N):
                self.vix_const[i, j] = self.xi(self.frak_t[i, j]) * ml(self.const_b * self.frak_t[i, j] ** (2 * self.H), self.beta) ** (-1) * ml(
                    self.const_b * (self.frak_t[i, j] - self.time_grid[i]) ** (2 * self.H), self.beta)

    def _cov_struct(self, t, s, T):
        '''
        Covariance
        '''
        first_part = hyp2f1(-self.H_minus, self.H_plus, 1 + self.H_plus,
                               -t / (s - t))
        second_part = hyp2f1(-self.H_minus, self.H_plus, 1 + self.H_plus,
                                (T - t) / (s - t))
        return (s - t) ** self.H_plus / self.H_plus * (
                    t ** self.H_plus * first_part - (t - T) ** self.H_plus * second_part)

    def _var_struct(self, t, T):
        '''
        Variance
        '''
        return (t ** (2 * self.H) - (t - T) ** (2 * self.H)) / (2 * self.H)

    def _corr_struct(self, t, s, T):
        '''
        Correlation
        '''
        return self._cov_struct(t, s, T) / np.sqrt(self._var_struct(t, T) * self._var_struct(s, T))

    @staticmethod
    @njit
    def _VIX(N, beta, vix_const, v_process, const, diff, Delta):
        '''
        Approximation of integral in VIX definition.
        '''
        vix = 0

        for j in range(N - 1):
            val1 = vix_const[j] * zeta(const * v_process[j], beta)

            val2 = vix_const[j + 1] * zeta(const * v_process[j + 1], beta)

            vix += 0.5 * (val1 + val2) * diff

        return (vix / Delta) ** 0.5

    def vix_pricing(self):
        '''
        VIX Pricer.
        '''
        self.vix_matrix = np.zeros((self.mc_sims, self.n_T - 1))

        # Variate generation
        variates = np.random.normal(0, 1, (self.mc_sims, self.cutoff))
        standard_normal_variates = np.random.normal(0, 1, (self.mc_sims, self.n_T - 1, self.N - self.cutoff))

        for l in tqdm(range(self.mc_sims)):
            for i in range(1, self.n_T):
                # Computing Volterra process on the first cutoff = 8 points
                self.v_process[l, :self.cutoff] = self.v_cov_matrix_chol[i].dot(variates[l, :])

                # Computing the Volterra process on the remaining points
                for j in range(self.cutoff, self.N):
                    self.v_process[l, j] = self.var[i, j - self.cutoff] * (
                            self.corr[i, j - self.cutoff] * self.v_process[l, j - 1] /
                            (self.var[i, j - 1 - self.cutoff]) + (1 - self.corr[i, j - self.cutoff] ** 2) ** 0.5 *
                            standard_normal_variates[l, i - 1, j - self.cutoff]
                            )

                # Compute VIX
                self.vix_matrix[l, i - 1] = self._VIX(self.N, self.beta, self.vix_const[i], self.v_process[l, :],
                                                      self.const, self.diff[i], self.Delta)

def j1(xi0, alpha, beta, eta):
    c_alpha = 1 / gamma(0.5 * (alpha + 1))
    delta = 1 / 12

    return xi0 * eta * c_alpha * np.pi ** 0.5 / 2 / gamma(1 + 0.5 * beta) * delta ** (alpha / 2 + 0.5) / (
                alpha / 2 + 0.5)

def j2(xi0, alpha, beta, eta):
    c_alpha = 1 / gamma(0.5 * (alpha + 1))
    delta = 1 / 12
    return xi0 * (eta * c_alpha) ** 2 / gamma(1 + beta) * delta ** (alpha) / (alpha)

def limTj3(xi0, alpha, beta, eta):
    c_alpha = 1 / gamma(0.5 * (alpha + 1))
    return -xi0 * (eta * c_alpha) ** 3 * np.pi ** 0.5 * 3 / 4 / gamma(1 + 1.5 * beta) / (3 * alpha / 2 - 0.5)

def VIXImpliedVolAsymptotics(xi0, alpha, beta, eta):
    '''
    gBergomi VIX Asymptotic Implied Volatility
    '''
    delta = 1 / 12
    return j1(xi0, alpha, beta, eta) / 2 / delta / xi0

def VIXSkewAsymptotics(xi0, alpha, beta, eta):
    '''
    gBergomi VIX Asymptotic Skew
    '''
    return j2(xi0, alpha, beta, eta) / j1(xi0, alpha, beta, eta) / 2 - VIXImpliedVolAsymptotics(xi0, alpha, beta, eta)

def VIXCurvatureAsymptotics(xi0, alpha, beta, eta):
    '''
    gBergomi VIX Asymptotic Curvature
    '''
    delta = 1 / 12
    return 2 * delta * xi0 / 3 / j1(xi0, alpha, beta, eta) ** 2 * limTj3(xi0, alpha, beta, eta)

def SPXSkewAsymptotics(alpha, beta, rho, eta):
    '''
    gBergomi SPX Asymptotic Skew
    '''
    c_alpha = 1 /gamma(0.5 * (alpha + 1))
    return np.pi ** 0.5 * c_alpha * rho * eta / ((alpha + 1) * (alpha + 3) * gamma(1 + 0.5 * beta))