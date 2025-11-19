import numpy as np
import matplotlib.pyplot as plt
import os

Q0 = 0
Q1 = 1
Q2 = 1
c = 2
t1 = 8
t2 = 17


t = np.linspace(0, 24, 1000)  # time in hours
H = 3 * np.sin(np.pi / 24 * t) + \
    1.5 * np.sin(4 * np.pi / 24 * t - np.pi/10) 

Q = Q0 + Q1*np.exp(-1/c*(t-t1)**2) + Q2*np.exp(-1/c*(t-t2)**2)

plt.plot(t,3 * np.sin(2*np.pi / 24 * t), label='H1')
plt.plot(t,1.5 * np.sin(4 * np.pi / 24 * t - np.pi/10), label = "H2")

# plt.plot(t, Q, label='Q(t)')
plt.plot(t,H,label= "sin")
plt.xlabel('Time (s)')
plt.legend()
plt.show()