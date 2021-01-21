from numpy import *

class EulerCallMonteCarlo:

    def __init__(self, S0,K,T,r,sgm,n):
        self.S0 = S0
        self.K = K
        self.sgm = sgm
        self.r = r
        self.T=T
        self.n = n

    def price(self):
        random.seed(20)
        n_steps=5000
        dt = self.T / n_steps
        S = zeros((n_steps, self.n))
        S[0] = self.S0

        for t in range(1, n_steps):
            # Draw random values to simulate Brownian motion
            Z = random.standard_normal(self.n)
            S[t] = S[t - 1] * exp((self.r - 0.5 * self.sgm ** 2) * dt + (self.sgm * sqrt(dt) * Z))
        C =  exp(-self.r * self.T) * 1 / self.n * sum(maximum(S[-1] - self.K, 0))
        return C