import numpy as np
import matplotlib.pyplot as plt

# Determination of Heat Exhanger parameters 
c_water = 4.18e3 
U = 4000  # Overall heat transfer coefficient [W/m2K] obtained via site https://www.engineeringtoolbox.com/heat-transfer-coefficients-exchangers-d_450.html

# Qdot = U * As * F * delta_T_mean
# For tapwater heater
Th_in = 65
Th_out = 35
Tc_in = 10
Tc_out = 60
Qdot = 31e3 # W
F = 0.95

dTmean = (Th_in - Tc_out) - (Th_out - Tc_in) / \
            np.log((Th_in - Tc_out)/(Th_out - Tc_in))

UAs = Qdot / (dTmean * F)
print(f'Tapwater: UAs = {UAs} W/K, U = 4000 W/m2K => As = {UAs/4000} m2')

# For spaceheating, but here you assume the same power as over the tap water HEX
Th_in = 65
Th_out = 35
Tc_in = 30
Tc_out = 60
F = 0.95

delta1 = Th_in - Tc_out
delta2 = Th_out - Tc_in

if delta1 == delta2: 
    dTmean = delta1
else:
    dTmean = (delta1 - delta2) / \
                np.log(delta1/delta2)

Qdot = UAs * (dTmean * F)
print(f'Spaceheating: transferred heat with tapwater parameters {Qdot}')

# Higher UAs needed for tapwater requires less flow for the space heating system. 
# But now the problem is that the 


def NTU_method(mflow_h):
 
    mflow_c = 0.15 # kg/s, max cold side flow rate 

    Th_in = 65
    Tc_in = 10
    F = 0.95

    # Heat capacity rates
    Cc = mflow_c * c_water
    Ch = mflow_h * c_water         

    Cmin = min(Cc, Ch)
    Cmax = max(Cc, Ch)
    Cr = Cmin / Cmax

    NTU = (UAs) / Cmin

    # Effectiveness calculation for counterflow heat exchanger
    if Cr != 1:
        epsilon = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
    else:
        epsilon = NTU / (1 + NTU)

    Q = F * epsilon * Cmin * (Th_in - Tc_in)

    Tc_out = Tc_in + Q / Cc
    Th_out = Th_in - Q / Ch
    
    return Tc_out, Th_out

mflowh_in = np.linspace(0.01,2,100)
Tc_out_array = []
Th_out_array = []   
for mflowh in mflowh_in:
    Tc_out, Th_out = NTU_method(mflowh)
    Tc_out_array.append(Tc_out)
    Th_out_array.append(Th_out)

plt.plot(mflowh_in, Tc_out_array, label='Tc_out')
plt.plot(mflowh_in, Th_out_array, label='Th_out')
plt.xlabel('Primairy side mass flow [kg/s]')
plt.ylabel('Outlet Temperature [C]')
plt.title('Heat Exchanger Outlet Temperatures vs Primary Side mass flow')
plt.legend()
plt.show()

# Tc_out, Th_out = NTU_method(0.027129160528221605*60)
# print(f'Tc_out {Tc_out}, Th_out {Th_out}')