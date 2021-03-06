from numpy import *
import matplotlib.pyplot as plt
class EulerCallMonteCarlo:

    def __init__(self, S0, K, T, r, sgm, steps, simulations):
        self.S0 = S0
        self.K = K
        self.sgm = sgm
        self.r = r
        self.T=T
        self.steps = steps
        self.simulations = simulations

    def price(self):
        random.seed(20)
        dt = self.T / self.steps
        S = zeros((self.steps, self.simulations))
        S[0] = self.S0

        for t in range(1, self.steps):
            Z = random.standard_normal(self.simulations)
            S[t] = S[t - 1] * exp((self.r - 0.5 * self.sgm ** 2) * dt + (self.sgm * sqrt(dt) * Z))
        C =  exp(-self.r * self.T) * 1 / self.simulations * sum(maximum(S[-1] - self.K, 0))
        plt.plot(linspace(0, 1, self.steps),  S, '-')
        plt.show()
        return C


euroCallEuler = EulerCallMonteCarlo(100,120,1,0.2,0.5,100,1000).price()
print("European Call Price Euler Method: {:.2f}".format(euroCallEuler))