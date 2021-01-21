import numpy as np

class LagrangePoly:
    def __init__(self, X, Y):
        self.n = len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def basis(self, x, j):
        b = [(x - self.X[m]) / (self.X[j] - self.X[m]) for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.Y[j]

    def interpolate(self, x):
        b = [self.basis(x, j) for j in range(self.n)]
        return np.sum(b, axis=0)

X  = [-0.7,0.7]
Y  = [6.6498,12.6826]


lp = LagrangePoly(X, Y)

y = lp.interpolate(np.linspace(-2.5, 2.5, 2))

print(y)
#cdf_y = scipy.stats.norm.cdf(xx)


# plt.scatter(Y,gamma.cdf(Y,5,scale=2) ,  c='k')
# plt.plot(y, gamma.cdf(y,5,scale=2), linestyle='-')
# plt.show()
