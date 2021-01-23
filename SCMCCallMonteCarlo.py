from numpy import  *
from SCMC import  SCMC
import matplotlib.pyplot as plt
class SCMCCallMonteCarlo:

 def __init__(self, S0,K,T,r,sgm,steps,simulations):
        self.S0 = S0
        self.K = K
        self.sgm = sgm
        self.r = r
        self.T=T
        self.steps = steps
        self.simulations = simulations

 def price(self):
        dt = self.T / self.steps
        S = zeros((self.steps, self.simulations))
        S[0] = self.S0

        for t in range(1, self.steps):
             #retrieving the interpolated values using Lagrage function
             Z = SCMC(self.r, self.sgm, self.simulations).calculateY()
             #Is  below correct way to implement the simulation?
             S[t] = S[t - 1] * exp((self.r - 0.5 * self.sgm ** 2) * dt + (self.sgm * sqrt(dt) * Z))

        plt.plot(linspace(0, self.T, self.steps), S, '-')
        plt.show()
        C = exp(-self.r * self.T) * 1 / self.simulations * sum(maximum(S[-1] - self.K, 0))
        return C

euroCallSCMC = SCMCCallMonteCarlo(100,120,1,0.2,0.5,10,10000).price()
print("European Call Price Euler Method: {:.2f}".format(euroCallSCMC))