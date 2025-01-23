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
        
        """
        # Initialize history
        pipe.bnode_init(dt, num_steps, v_flow, T_inlet)
        # self.bnode_init(min(v_flow), T_inlet[0]) # NOTE: maybe use the minimum velocity as initial velocity
               
        # Run simulation with the extended arrays
        for N in range(num_steps):
            pipe.bnode_method(N, T_ambt)
            
        # Get rid of the history values TODO: need to think of a more elegant way to fix this.
        pipe.T = pipe.T[pipe.hist_len:]
        pipe.T_lossless = pipe.T_lossless[pipe.hist_len:]
        pipe.T_cap = pipe.T_cap[pipe.hist_len:]
        
    def simulate_network_thermodynamics(self, 
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
        
        pass 

    def plot_results_single_pipe_simulation(self, time, T_inlet, pipe, v_flow, decimal = 4):
        """
        Plot the results of the simulation
        time: time array for the simulation
        T_inlet: inlet temperature array
        pipe: pipe object with simulation results
        v_flow: flow velocity array
        decimal: number of decimals to round the temperature arrays to
        """
        plt.figure(figsize=(10, 6))
        plt.title("Water temperature")
        plt.ticklabel_format(style='plain', axis='y')  # Use plain formatting for y-axis


        plt.plot(time, T_inlet, label='Inlet Temperature')
        plt.plot(time, np.round(pipe.T_lossless, decimal), label = "Lossless temperature")
        plt.plot(time, np.round(pipe.T_cap, decimal), label='Temperature with pipe capacity')
        plt.plot(time, np.round(pipe.T, decimal), label = 'Real temperature')
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





# Example test case
if __name__ == "__main__":

    # Pipe parameters
    pipe_radius_outer = 0.1 # m DUMMY
    pipe_radius_inner = 0.08 # m DUMMY
    K = 0.4 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
    pipe_data = [pipe_radius_outer, pipe_radius_inner, K]
      
    # Create network
    net = Network()
    net.add_node("Node 1", 0.0, 0.0, 0.0)
    net.add_node("Node 2", 1.0, 2.0, 30.0)
    net.add_pipe("Pipe 1", "Node 1", "Node 2", pipe_data)
   
    # Ambient temperature
    T_ambt = 15 # [C]

    # Time parameters
    dt = 1  # time step
    total_time = 300 # sec
    time = np.arange(0, total_time + dt, dt) # time array
    num_steps = len(time) 

    # Inlet temperature
    T_inlet = 80 + 5 * np.sin(np.linspace(0, 2*np.pi, num_steps))   # Oscillating inlet temperature
    # T_inlet = np.ones(num_steps) * 80                          # Constant
    # T_inlet = 80 + 1* square(2 * np.pi * time / 20)                 # Square wave with a period of 20 steps
    
    # Flow velocity
    v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, num_steps)) # Oscillating flow velocity
    # v_flow = np.ones(num_steps) * 2                           # Constant
    # v_flow = 1.5 + 0.5 * square(2 * np.pi * time / 50)        # Square wave flow velocity, 50 is the period
 
    # Run simulation
    sim = Simulation()
    pipe = (net.pipes['Pipe 1'])['pipe_class']
    sim.simulate_pipe_temperature(pipe, dt, num_steps, T_ambt, T_inlet, v_flow)
     
    # Plot results
    sim.plot_results_single_pipe_simulation(time, T_inlet, pipe, v_flow)

    # net.nodes['Node 2'].set_T() # For now it points to a whole array not to one single timestep. 
    # print(net.nodes['Node 2'].get_T())
    
    # Start interactive session
    # import code
    # code.interact(local=locals())