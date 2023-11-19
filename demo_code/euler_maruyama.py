
'''
Euler Maruyama approximations
https://gist.github.com/kbelcher3/5d02f2da5d9dc3566c292695babb219f
'''

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
num_sims = 1

# Number of points in partition
N = 1500

# Initial value
X_0 = 1 
Y_0 = X_0

# Starting time
t_0 = 0

# Ending time
T = 10

# SDE for GBM: dX_t = r_0*X_t dt + sigma_0*X_t dB_t
r_0 = 0
sigma_0 = 1
	
# Time increments
dt = float(T - t_0) / N

# Times 
t = np.arange(t_0, T, dt) 

# Brownian increments
dB = np.zeros(N)
dB[0] = 0

# Brownian samples
B = np.zeros(N)
B[0] = 0

# Simulated process
X    = np.zeros(N) 
X[0] = 1

# Approximated process
Y    = np.zeros(N)
Y[0] = Y_0

# Sample means across all simulations
SX = np.zeros(N)
SY = np.zeros(N)

# Iterate
for n in range(num_sims):
    for i in range(1, t.size):
        # Generate dB_t
        dB[i] = np.random.normal(loc = 0.0, scale = np.sqrt(dt))
        
        # Generate B_t
        B[i] = np.random.normal(loc = 0.0, scale = np.sqrt(t[i]) )
        
        # Simulate (blue)
        X[i] = X_0 * np.exp( (r_0 - 0.5 * sigma_0*sigma_0)*(i * dt) + (float(sigma_0) * B[i] ))
        SX[i] = SX[i] + X[i]/num_sims
        
        # Approximate (green)
        Y[i] = Y[i-1] + (r_0 * Y[i-1]) * dt + (sigma_0 * Y[i-1]) * dB[i]
        SY[i] = SY[i] + Y[i]/num_sims
    
# Plot
plt.plot(t, SX)
plt.plot(t, SY)    
plt.show()