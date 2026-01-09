import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

num_steps = 1000
t = np.linspace(0, 24, num_steps)  # time in hours

# Fit the function from Max
xfit = [0,4,8,12,13,16,18,20,24]
yfit = [0.31, 0.43, 0.65, 0.52, 0.5,0.53,0.6,0.51,0.31]

x = np.asarray(xfit)
y = np.asarray(yfit)

def two_sin(x, A1, w1, p1, A2, w2, p2, offset):
    return A1 * np.sin(w1 * x + p1) + A2 * np.sin(w2 * x + p2) + offset

# initial guesses: amplitudes, angular frequencies (rad/h), phases, offset
p0 = [0.4, np.pi/12, 0.0, 0.2, 2.8 * np.pi / 24, -0.3, 0.0]
bounds = ([-5, 0.0, -4*np.pi, -5, 0.0, -4*np.pi, -2],
          [ 5, 2*np.pi,  4*np.pi,  5, 2*np.pi,  4*np.pi,  2])

popt, pcov = curve_fit(two_sin, x, y, p0=p0, bounds=bounds, maxfev=200000)
A1, w1, p1, A2, w2, p2, offset = popt

print(f'Fitted parameters:\nA1={A1}, w1={w1}, p1={p1}\nA2={A2}, w2={w2}, p2={p2}\noffset={offset}')

# build fitted curve on the existing fine time grid `t`
y_fit_full = two_sin(t, *popt)

# definite integral (area under the fitted curve) over the t grid
total_area = np.trapz(3600*y_fit_full, t)  # units: y * hours
print(f"Integrated area (0-{t[-1]} h): {total_area:.6f} MJ")

# To get the heat demand for one house, divide by total_area/65*1000  to get kW
correction_factor = 1000 / (total_area/65)
print(f'Correction factor: {correction_factor}')
demand_one_house = correction_factor*y_fit_full
plt.figure()
plt.plot(t,demand_one_house,label='Heat demand 1 house (kW)')
plt.ylabel('Heat demand (kW)')
plt.xlabel('Time (h)')
plt.grid()
plt.title(f'Heat demand profile 1 appartement, total = {np.round(3.6*np.trapz(demand_one_house,t),2)} MJ')

plt.figure()
plt.plot(x, y, 'o', label='data')
plt.plot(t, y_fit_full, '-', label='two-sin fit')
plt.plot(t, A1 * np.sin(w1 * t + p1), '--', label='component 1')
plt.plot(t, A2 * np.sin(w2 * t + p2), '--', label='component 2')
plt.xlabel('Time (h)')
plt.legend()
plt.grid()
plt.show()


# def smooth_peak(x, x0, sigma, A=1):
#     return A * np.exp(-(x - x0)**2 / (2*sigma**2))

# # Create a time vector for one day (24 hours) with 1-minute intervals
# t = np.linspace(0, 24*3600, 24*3600)  # time in seconds

# # Create a heat demand profile with two peaks (morning and evening)
# y_pre_fit = smooth_peak(t, 8*3600, 0.07*3600) 

# # + smooth_peak(t, 19*3600, 0.15*3600)  # units: W

# total_area = np.trapz(y_pre_fit, t)  # units: y * hours

# scaling_factor = 18e6 / (total_area)  # scaling factor to reach 65 MJ
# y = y_pre_fit * scaling_factor  # scaled heat demand profile
# plt.plot(t, y)
# plt.title("One day heat demand profile (scaled to 18 MJ)")
# plt.xlabel("Time (hours)")
# plt.ylabel("Heat Demand (W)")

# ax = plt.gca()
# ax.set_xlim(0, 24 * 3600)  # limits in seconds
# ticks_seconds = np.arange(0, 25, 4) * 3600
# ax.set_xticks(ticks_seconds)
# ax.set_xticklabels([f'{int(h)}' for h in np.arange(0, 25, 4)])

# plt.grid()
# plt.show()

# print(f'Total energy demand over the day: {np.trapz(60*1e3*y, t)/1e6} MJ')  # Convert kW*min to MJ