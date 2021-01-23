from Lagrange import LagrangePoly
import numpy as np
from scipy.stats import norm
class SCMC:
    def __init__(self, mean, variance, simulations):
        self.mean = mean
        self.variance = variance
        self.simulations = simulations

    def calculateY(self):
        # Step 1 - chose 5 - as paper 21 it gives optimal results
        x = np.polynomial.hermite_e.hermegauss(5)[0]
        # Step 2
        y = norm.ppf(norm.cdf(x), loc=self.mean, scale=self.variance)
        # Step 3
        # determine the interpolation function
        lp = LagrangePoly(x, y)
        # Step 4
        # Generate sample from X
        samplex = np.random.standard_normal(self.simulations)
        # interpolate the 1000 values
        return lp.interpolate(samplex)

#print(SCMC(0.05,0.03,1000).calculateY())
#print(np.random.standard_normal(10))