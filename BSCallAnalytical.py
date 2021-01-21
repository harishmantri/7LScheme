import numpy as np
import scipy.stats as si


class BSCallAnalytical:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        result = (self.S * si.norm.cdf(d1, 0.0, 1.0) - self.K *
                  np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0))
        return result
