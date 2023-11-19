'''
Geometric Brownian Motion: Simulating Stock Price Dynamics by John Sullivan
https://jtsulliv.github.io/stock-movement/

'''


''' Necessary Packages'''
import numpy as np
import matplotlib.pyplot as plt

''' Clear '''
plt.close('all');


seed = 1
N = 1000  # increments


def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1. / N  # time step
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)  # brownian path
    return W, b, dt


# brownian increments
b = Brownian(seed, N)[1]

# brownian motion
W = Brownian(seed, N)[0]
W = np.insert(W, 0, 0.)


# brownian increments
plt.rcParams['figure.figsize'] = (10,8)
xb = np.linspace(0, len(b), len(b))
plt.plot(xb, b)
plt.title('Brownian Increments')
plt.show()


# Standard Brownian Motion
xw = np.linspace(0, len(W), len(W))
plt.plot(xw, W)
plt.title('Standard Brownian Motion')



# GBM Exact Solution

def GBM(So, mu, sigma, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

So = 55.25 #initial stock price
mu = 0.15 #returns (drift coefficient)
sigma = 0.4 #volatility (diffusion coefficient)
W = Brownian(seed, N)[0] #brownian motion
T = N+1


soln = GBM(So, mu, sigma, W, T, N)[0]    # Exact solution
t = GBM(So, mu, sigma, W, T, N)[1]       # time increments for  plotting

plt.plot(t, soln)
plt.ylabel('Stock Price, $')
plt.title('Geometric Brownian Motion')