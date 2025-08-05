import numpy as np
import matplotlib.pyplot as plt

# Input restricted to [-1, 1]
u = np.linspace(-1, 1, 800)

# Activation functions
relu = np.maximum(u, 0)
rect_sq = relu**2
alpha = 1.0
div_norm = rect_sq / (1 + alpha * rect_sq)
tanh = np.tanh(u)

# Plot
plt.figure(figsize=(6,4))
plt.plot(u, rect_sq, label=r'Rectify-square $[u]_+^2$')
plt.plot(u, div_norm, label=r'Divisive norm $\frac{[u]_+^2}{1+\alpha[u]_+^2}$')
plt.plot(u, tanh, label=r'$\tanh(u)$')
plt.xlabel(r'Input $u$')
plt.ylabel('Output')
plt.title('Activation functions (input restricted to $[-1,1]$)')
plt.xlim(-1, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()