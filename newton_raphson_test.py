import numpy as np

rho_w = 1000 # [kg/m3]
D =  0.025*2
epsilon = 0.000045 # [m] for commercial steel pipe
Re = 4e4  # example Reynolds number

log_term = (epsilon/D)/3.7 + (6.9/Re)**1.11
f = (1 / (-1.8 * np.log10(log_term)))**2
pump_pressure = 1e5  # [Pa] = 1 bar

print(f'pressure coefficient = {f}')

def pressure_df(L):
    return f * 8 * rho_w * L / (np.pi**2 * D**5)

loop_matrix = np.array([[ 1,  1,  1,  1, 0, 0, 0],
                        [0, 0, -1, 0, 1, 1, 1]])


incidence_matrix = np.array([
                            [0,1,-1,0,-1,0,0],
                            [0,0,0,0,1,-1,0],
                            [0,0,0,0,0,1,-1],
                            [0,0,1,-1,0,0,1],
                            [-1,0,0,1,0,0,0]]
                            )




df2 = pressure_df(2)
df3 = pressure_df(3)

k_vector = np.array([df3,df2,df3,df2,df2,df3,df2])
flow_init = np.array([0.1,0.1,0.05,0.1,0.05,0.05,0.05])


# newton-raphson
error = 0
tolerance = 1e-6
flow = flow_init.copy()
max_iter = 100

for it in range(max_iter):

    head_loss = np.matmul(loop_matrix, k_vector * flow**2)
    continuity = np.matmul(incidence_matrix,flow)
    F = np.concatenate([continuity, head_loss], axis = 0) + np.array([0,0,0,0,0,0,pump_pressure,0]) 
    error = np.linalg.norm(F)
    if error < tolerance:
        break

    J = np.concatenate([incidence_matrix,loop_matrix * (2 * k_vector * flow)])  # Jacobian (2x7)
    print(f'determinant Jacobian {np.linalg.det(J)}')
    delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
    flow += delta

print(f'flow {flow}, iterations {it+1}, error {error}')
if error >= tolerance:
    print("Warning: Newton-Raphson did not converge within", max_iter, "iterations")


