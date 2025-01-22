import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from typing import Union

from node import Node
from pipe import Pipe
from network import Network

class Simulation:
    
    def simulate_pipe_temperature(self,
                                pipe: Pipe,
                                dt : float, 
                                num_steps : int, 
                                T_ambt : float,
                                T_inlet : np.ndarray[Union[float]],
                                v_flow : np.ndarray[Union[float]]):
        """
        Simulate temperature dynamics for a pipe section
                
        Args:
        T_inlet: array of inlet temperatures
        v_flow: array of flow velocities
        num_steps: number of simulation steps
        
        Returns:
        T: array of out temperatures
        """
        # Initialize history
        pipe.bnode_init(dt, num_steps, T_ambt, v_flow[0], T_inlet[0])
        # self.bnode_init(min(v_flow), T_inlet[0]) # NOTE: maybe use the minimum velocity as initial velocity
        
        # Combine history with actual data
        v_extended = np.concatenate([pipe.v_history, v_flow])
        T_extended = np.concatenate([pipe.T_history, T_inlet])
        
        pipe.T_lossless_out = np.zeros(num_steps)
        pipe.T_cap_out  = np.zeros(num_steps)
        pipe.T = np.zeros(num_steps)

        hist_len = len(pipe.v_history)
        
        # Run simulation with the extended arrays
        for N in range(num_steps):
            actual_N = N + hist_len  # Adjust index to account for history
            pipe.T_lossless_out[N], pipe.T_cap_out[N], pipe.T[N] = pipe.bnode_method(v_extended[:actual_N+1], 
                                    T_extended[:actual_N+1], 
                                    actual_N)
        
    def simulate_network_thermo(self, 
                         network : Network, 
                         dt : float, 
                         time : int,
                         T_in : np.ndarray[Union[float]]) -> None:
        """
        Simulate temperature dynamics for a network.
        
        Args:
        network: the network to be simulated
        dt: time step
        time: total time of simulation
        """
        






# Example test case
if __name__ == "__main__":

    # Pipe parameters
    pipe_radius_outer = 0.1 # m DUMMY
    pipe_radius_inner = 0.08 # m DUMMY
    K = 0.4 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
    pipe_data = [pipe_radius_outer, pipe_radius_inner, K]
    
    # Time parameters
    dt = 1  # time step
    total_time = 1000 # sec
    time = np.arange(0, total_time + dt, dt) # time array
    num_steps = len(time) 
    
    # Create network
    net = Network()
    net.add_node("Node 1", 1.0, 2.0, 300)
    net.add_node("Node 2", 1.0, 2.0, 5.0)
    net.add_pipe("Pipe 1", "Node 1", "Node 2", pipe_data)
    # net.add_pipe("Pipe 2", "Node 2", "Node 3")
    

    # Ambient temperature
    T_ambt = 15 # [C]

    pipe = (net.pipes['Pipe 1'])['pipe_class']

    # Inlet temperature
    # T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, num_steps))   # Oscillating inlet temperature
    T_inlet = np.ones(pipe.num_steps) * 80                          # Constant
    # T_inlet = 80 + 1* square(2 * np.pi * time / 20)                 # Square wave with a period of 20 steps
    
    # Flow velocity
    v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, num_steps)) # Oscillating flow velocity
    # v_flow = np.ones(num_steps) * 2                           # Constant
    # v_flow = 1.5 + 0.5 * square(2 * np.pi * time / 50)        # Square wave flow velocity, 50 is the period
 
    # Run simulation
    sim = Simulation()
    sim.simulate_pipe_temperature(pipe, dt, num_steps, T_ambt, T_inlet, v_flow)
    
    net.nodes['Node 2'].set_T() # For now it points to a whole array not to one single timestep. 
    print(net.nodes['Node 2'].get_T())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.title("Water temperature")
    plt.plot(time, T_inlet, label='Inlet Temperature')
    plt.plot(time, pipe.T_lossless_out, label = "Lossless temperature")
    plt.plot(time, pipe.T_cap_out, label='Temperature with pipe capacity')
    plt.plot(time, pipe.T, label = 'Real temperature')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(pipe.t_stay_array)
    plt.title('Average delay in the pipe')

    plt.figure()
    plt.plot(v_flow)
    plt.title('Flow velocity')

    plt.show()

    # Start interactive session
    # import code
    # code.interact(local=locals())