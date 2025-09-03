

import numpy as np
import matplotlib.pyplot as plt

# --- Euler–Maruyama integration ------------------------------------------
for i in range(1, length):
    y[i] = (
        y[i - 1]
        + drift(y[i - 1], i * dt) * dt
        + diffusion(y[i - 1], i * dt) * noise[i]
    )

# --- Plot the result ------------------------------------------------------
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Ornstein–Uhlenbeck sample path (Euler–Maruyama)")
plt.show()

