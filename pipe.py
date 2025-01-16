import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

class Pipes:

    def __init__(
            self, 
            pipe_length : float, 
            pipe_radius_outer : float,
            pipe_radius_inner : float,
            dt : float,
            num_steps : int,
            K : float,
            T_ambt : float):
        """
        Initialize pipe.

        Parameters: 
        pipe_length [m]
        pipe_radius_outer [m]
        pipe_radius_inner [m]
        dt: time step [t]
        num_steps: number of simulation steps 
        K: overall heat transmission coefficient [W/m2K]
        T_ambt: ambient temperature [C]

        #TODO: update
        """

        self.L = pipe_length
        self.radius_outer = pipe_radius_outer 
        self.radius_inner = pipe_radius_inner

        self.dt = dt
        self.num_steps = num_steps

        # Physical constants
        self.K = K
         
        self.rho_water = 1e3 # [kg/m3] NOTE: maybe later make a constant file. When there are too many constants.
        self.rho_pipe = 8e3 #  [kg/m3] 
        self.c_water = 4.18e3 # [J/kg K] specific heat capacity
        self.c_pipe = 0.466e3 # [J/kg K] specific heat capacity of steel NOTE: DUMMY checked on wiki, needs to be more accurate 
        
        self.T_ambt = T_ambt # [C] ambient temperature 
        self.C_pipe = np.pi * (self.radius_outer ** 2 - self.radius_inner ** 2) * self.rho_pipe * self.c_pipe * self.L # [J/K] total heat capacity pipe
        self.inner_cs = np.pi * self.radius_inner ** 2 # inner cross section area
        self.outer_cs = np.pi * self.radius_outer ** 2 # outer cross section area

    def bnode_init(self, v_init, T_init):
        """
        Initialize history of velocities and temperatures to ensure valid solutions.
        
        This should only be necessary for the initial use of the node method. As later it should remember its previous temperatures. 
        
        Parameters:
        v_init: initial velocity
        T_init: initial temperature
        
        Returns:
        v_history: array of historical velocities
        T_history: array of historical temperatures
        """
        # Calculate minimum history length needed based on pipe length and flow velocity
        min_steps = int(np.ceil((self.L + v_init * self.dt) / (v_init * self.dt)))
        # Add some margin to ensure we have enough history NOTE: based on what?
        history_length = min_steps + 5
        print(f' HISTORY LENGTH {history_length}')
        
        # Initialize velocity and temperature of water 
        self.v_history = np.ones(history_length) * v_init
        self.T_history = np.ones(history_length) * T_init

        # initialize temperature for the pipe
        self.T_pipe = np.ones(history_length + self.num_steps) * T_init  # NOTE: maybe use a different initialization, but for now it is good. It is longer than it should be but it works better with actual_N

        self.t_stay_array = np.ones(self.num_steps) # debug 
        self.first_term_array = np.zeros(history_length + self.num_steps) # debug
        self.second_term_array = np.zeros(history_length + self.num_steps) # debug
        self.third_term_array = np.zeros(history_length + self.num_steps) # debug


    def bnode_method(self, v, T_k, N):
        """
        Implementation of b-node method for temperature dynamics in pipes
        based on Benonysson (1991)
        
        Parameters:
        v: array of flow velocities at each time step k (including history)
        T_k: array of temperatures at inflow end at each time step k (including history)
        N: current time step number
        
        Returns:
        T_N: Temperature of water flowing from the pipe at time step N
        """
        # Find smallest integer n and corresponding R
        n = 0
        while True:
            R = sum(v[N-n:N+1] * self.dt)
            if R > self.L:
                break
            n += 1
        
        # Find smallest integer m
        m = 0
        while True:
            sum_term = sum(v[N-m:N+1] * self.dt)
            if sum_term > self.L + v[N] * self.dt:
                break
            m += 1
        
        # Compute Y and S
        Y = sum(v[N-m+1:N-n] * T_k[N-m+1:N-n] * self.dt)
        
        # Calculate S based on conditions
        if m > n:
            S = sum(v[N-m+1:N+1] * self.dt)
        else:  # m == n
            S = R
        
        #debug print(R, Y, S)

        # Compute the lossless output temperature T_N
        T_N_lossless = ((R - self.L) * T_k[N-n] + Y + (v[N] * self.dt - S + self.L) * T_k[N-m]) / (v[N] * self.dt)
        
        # Take into account the heat capacity of the steel pipe
        m_flow = v[N] * self.rho_water * self.inner_cs  # mass flow 
        
        T_N_pipe = (m_flow * self.c_water * T_N_lossless * self.dt + self.C_pipe * self.T_pipe[N-1]) / (self.C_pipe + m_flow * self.c_water * self.dt)

        self.T_pipe[N] = T_N_pipe # update temperature pipe

        # determine average delay in the pipe
        t_stay = self.average_delay(n,m,N,R,S,v)
        self.t_stay_array[N-len(self.v_history)] = t_stay
        print(f't_stay {t_stay}') #debug 

        T_real_out = self.T_ambt + (T_N_pipe - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section

       
        return T_N_lossless, T_N_pipe, T_real_out #debug 

    def average_delay(self,n,m,N,R,S,v):
        """    
        Source: Maurer. J, Comparison of discrete dynamic pipeline models for operational optimization of District Heating Networks. 2021

        Parameters:
        n : The number of complete pipe lengths traversed.
        m : The number of time steps for the final partial pipe length.
        N : Total number of time steps.
        R : The remaining distance at the start of the calculation.
        S : The remaining distance at the end of the calculation.
        v : Array of velocities at each time step.
        
        Returns:
        The average delay time of the water in the pipe.
        """

        first_term = n * (R - self.L)

        second_term = 0
        for i in range(N-m+1, N-n):
            second_term += (N - i) * v[i] * self.dt

        third_term = m * (v[N] * self.dt + (self.L - S))

        print(f' n = {n} and m = {m}')

        return (first_term + second_term + third_term)/v[N]

    def simulate_pipe_temperature(self, T_inlet, v_flow, num_steps):
        """
        Simulate temperature dynamics for a pipe section
                
        Parameters:
        T_inlet: array of inlet temperatures
        v_flow: array of flow velocities
        num_steps: number of simulation steps
        
        Returns:
        T_out: array of out temperatures
        """
        # Initialize history
        self.bnode_init(v_flow[0], T_inlet[0])
        # self.bnode_init(min(v_flow), T_inlet[0]) # NOTE: maybe use the minimum velocity as initial velocity
        
        # Combine history with actual data
        v_extended = np.concatenate([self.v_history, v_flow])
        T_extended = np.concatenate([self.T_history, T_inlet])
        
        self.T_lossless_out = np.zeros(self.num_steps)
        self.T_cap_out  = np.zeros(self.num_steps)
        self.T_real_out = np.zeros(self.num_steps)

        hist_len = len(self.v_history)
        
        # Run simulation with the extended arrays
        for N in range(num_steps):
            actual_N = N + hist_len  # Adjust index to account for history
            self.T_lossless_out[N], self.T_cap_out[N], self.T_real_out[N] = self.bnode_method(v_extended[:actual_N+1], 
                                    T_extended[:actual_N+1], 
                                    actual_N)
            
    def thermal_transmission_coef(self):
        """
        Calculate the thermal transmission coefficient of the pipe.
        TODO: Implement this function
        """
        pass




# Example test case
if __name__ == "__main__":

    # Pipe parameters
    Z = 30.0  # pipe length
    pipe_radius_outer = 0.1 # m DUMMY
    pipe_radius_inner = 0.08 # m DUMMY
    K = 100 # heat transmission coefficient DUMMY 
    
    # Time parameters
    dt = 1  # time step
    total_time = 100 # sec
    time = time_array = np.arange(0, total_time + dt, dt)
    num_steps = len(time)
    
    # Ambient temperature
    T_ambt = 15 # [C]

    pipe = Pipes(Z,pipe_radius_outer,pipe_radius_inner,dt,num_steps,K, T_ambt)
   
    # Inlet temperature
    # T_inlet = 20 + 5 * np.sin(np.linspace(0, 2*np.pi, num_steps))  # Oscillating inlet temperature
    # T_inlet = np.ones(pipe.num_steps) * 20 #constant
    T_inlet = 20 + 1* square(2 * np.pi * time / 20)  # Square wave with a period of 20 steps
    
    # Flow velocity
    v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, pipe.num_steps))  # Oscillating flow velocity
    # v_flow = np.ones(num_steps) * 2  # constant
    # v_flow = 1.5 + 0.5 * square(2 * np.pi * time / 50)  # Square wave flow velocity, 50 is the period
 
    # Run simulation
    pipe.simulate_pipe_temperature(T_inlet, v_flow, num_steps)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.title("Water temperature")
    plt.plot(time, T_inlet, label='Inlet Temperature')
    plt.plot(time, pipe.T_lossless_out, label = "Lossless temperature")
    plt.plot(time, pipe.T_cap_out, label='Pipe temperature')
    plt.plot(time, pipe.T_real_out, label = 'Real temperature')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(pipe.t_stay_array)
    plt.title('Average delay in the pipe')

    # plt.figure()
    # plt.plot(pipe.first_term_array)
    # plt.title('First term')

    # plt.figure()
    # plt.plot(pipe.second_term_array)
    # plt.title('Second term')

    # plt.figure()
    # plt.plot(pipe.third_term_array)
    # plt.title('Third term')

    plt.figure()
    plt.plot(v_flow)
    plt.title('Flow velocity')

    plt.show()

    # Start interactive session
    # import code
    # code.interact(local=locals())
    
# TODO: validate the model with real data