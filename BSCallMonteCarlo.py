import  numpy as np
import matplotlib.pyplot as plt
class BSCallMonteCarlo:

    def __init__(self, S,K,T,r,sgm,n):
        self.S = S
        self.K = K
        self.sgm = sgm
        self.r = r
        self.T=T
        self.n = n

    def price(self):
       w = np.random.standard_normal(self.n)
       ST= self.S*np.exp((self.r-0.5*self.sgm**2)*self.T+self.sgm*np.sqrt(self.T)*w)
       payoff=ST-self.K
       payoff = payoff*(payoff>0)
       MC=np.exp(-self.r*self.T)*np.mean(payoff)
       return MC

#euro = BSCallMonteCarlo(100, 100, 1, 0.05, 0.03, 1000)
euro = BSCallMonteCarlo(1, 0.5, 4, 0.1, 0.05, 1000)
print(euro.price())