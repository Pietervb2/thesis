import numpy as np

dp = 726 # Pa
L = 3 # m 
rho = 1e3 # kg/m3
f = 0.0411 # Darcy friction factor
D = 0.04 # m



mflow = np.sqrt(dp / (8 * f * L / (np.pi**2 * D**5 * rho)))
print(f' mflow {mflow}')
print(f' velocity {mflow / (rho * np.pi * (D/2)**2)}')