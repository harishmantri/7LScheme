import  numpy as np

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
       payoff=(ST-self.K)
       MC=np.exp(-self.r*self.T)*np.mean(payoff)
       return MC