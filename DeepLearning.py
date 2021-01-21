import numpy as np
import scipy.stats as si
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend

def custom_activation(x):
    return backend.exp(x)

def euro_vanilla(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    result = (S * si.norm.cdf(d1, 0.0, 1.0) - K *
                  np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return result


random.seed(42)
draws = 300000
S = np.random.rand(draws) * 100
K = (np.random.randint(50, 150, draws) * .01) * S
T = np.random.randint(10, 300, draws) / 100
r = np.random.randint(1, 1000, draws) / 10000
sigma = np.random.randint(1, 50, draws) / 100
# generate option prices
opt_price = []
for i in range(draws):
    p = euro_vanilla(S[i], K[i], T[i], r[i], sigma[i])
    opt_price.append(p)
# create a dataframe
options = pd.DataFrame({'S': S,
                        'K': K,
                        'T': T,
                        'r': r,
                        'sigma': sigma,
                        'price': opt_price}
                       )

options['S'] = options['S']/options['K']
options['price']  = options['price'] /options['K']

X = options[['S', 'T', 'sigma', 'r']]
y = options[['price']]

n=  len(options)
n_train =  (int)(0.8 * n)
train = options[0:n_train]
X_train = train[['S', 'T', 'sigma', 'r']].values
y_train = train['price'].values


test = options[n_train+1:n]
X_test = test[['S', 'T', 'sigma', 'r']].values
y_test = test['price'].values

model = Sequential()


nodes = 120
model.add(Dense(nodes, input_dim=X_train.shape[1]))
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation(custom_activation))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=2)

import numpy as np
import matplotlib.pyplot as plt

def CheckAccuracy(y, y_hat):
    stats = dict()
    stats['diff'] = y - y_hat
    stats['mse'] = np.mean(stats['diff'] ** 2)
    print
    "Mean Squared Error:      ", stats['mse']

    stats['rmse'] = np.sqrt(stats['mse'])
    print
    "Root Mean Squared Error: ", stats['rmse']

    stats['mae'] = np.mean(abs(stats['diff']))
    print
    "Mean Absolute Error:     ", stats['mae']

    stats['mpe'] = np.sqrt(stats['mse']) / np.mean(y)
    print
    "Mean Percent Error:      ", stats['mpe']

    # plots
    #mpl.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14, 10))
    plt.scatter(y, y_hat, color='black', linewidth=0.3, alpha=0.4, s=0.5)
    plt.xlabel('Actual Price', fontsize=20, fontname='Times New Roman')
    plt.ylabel('Predicted Price', fontsize=20, fontname='Times New Roman')
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.hist(stats['diff'], bins=50, edgecolor='black', color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()

    return stats

y_train_hat = model.predict(X_train)
y_train_hat = np.squeeze(y_train_hat)

CheckAccuracy(y_train, y_train_hat)


y_test_hat = model.predict(X_test)
y_test_hat = np.squeeze(y_test_hat)

test_stats = CheckAccuracy(y_test, y_test_hat)
