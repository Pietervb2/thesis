import numpy as np
import matplotlib.pyplot as plt


class Pipes:

    def __init__(
            self, 
            pipe_length : float, 
            pipe_radius_outer : float,
            pipe_radius_inner : float,
            dt : float,
            num_steps : int,
            K : float):
        """
        Initialize pipe.

        Parameters: 
        pipe_length [m]
        pipe_radius [m]
        dt: time step [t]
        num_steps: number of simulation steps 
        K: overall heat transmission coefficient [W/m2K]

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
        
        self.T_ambt = 10 # [C] ambient temperature NOTE: constant
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
        # Add some margin to ensure we have enough history
        history_length = min_steps + 5
        
        # Initialize velocity and temperature of water 
        self.v_history = np.ones(history_length) * v_init
        self.T_history = np.ones(history_length) * T_init

        # initialize temperature for the pipe
        self.T_pipe = np.ones(history_length + self.num_steps) * T_init  # NOTE: maybe use a different initialization, but for now it is good. It is longer than it should be but it works better with actual_N
        

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

        # Compute the output temperature T_N
        T_N_lossless = ((R - self.L) * T_k[N-n] + Y + (v[N] * self.dt - S + self.L) * T_k[N-m]) / (v[N] * self.dt)
        
        # Take into account the heat capacity of the steel pipe
        m_flow = v[N] * self.rho_water * self.inner_cs  # mass flow 
        
        T_N_pipe = (m_flow * self.c_water * T_N_lossless * self.dt + self.C_pipe * self.T_pipe[N-1]) / (self.C_pipe + m_flow * self.c_water * self.dt)

        self.T_pipe[N] = T_N_pipe # NOTE: maybe use here the final temperature of the node

        # determine average delay in the pipe
        t_stay = self.average_delay(n,m,N,R,S,v,m_flow)

        T_real_out = self.T_ambt + (T_N_pipe - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section

       
        return T_N_lossless, T_N_pipe, T_real_out #debug 

    def average_delay(self,n,m,N,R,S,v,m_flow):

        first_term = n * (R - self.rho_water * self.L * self.inner_cs ) 
        
        second_term = 0
        for i in range(N-m+1,N-n):
            m_flow = v[i] * self.rho_water * self.inner_cs  # mass flow 
            second_term += (N - i) * m_flow * self.dt

        third_term = m * (m_flow * self.dt + self.rho_water * self.inner_cs * self.L - S)
       
        return (first_term + second_term + third_term)/m_flow

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
        
        # Combine history with actual data
        v_extended = np.concatenate([self.v_history, v_flow])
        T_extended = np.concatenate([self.T_history, T_inlet])
        
        self.T_lossless_out = np.zeros(num_steps)
        self.T_pipe_out  = np.zeros(num_steps)
        self.T_real_out = np.zeros(num_steps)

        hist_len = len(self.v_history)
        
        # Run simulation with the extended arrays
        for N in range(num_steps):
            actual_N = N + hist_len  # Adjust index to account for history
            self.T_lossless_out[N], self.T_pipe_out[N], self.T_real_out[N] = self.bnode_method(v_extended[:actual_N+1], 
                                    T_extended[:actual_N+1], 
                                    actual_N)
          

# Example test case
if __name__ == "__main__":
    # Set up parameters
    Z = 30.0  # pipe length
    dt = 1.0   # time step
    pipe_radius_outer = 0.1 # m DUMMY
    pipe_radius_inner = 0.08 # m DUMMY
    num_steps = 200
    K = 1 # heat transmission coefficient DUMMY 
    pipe = Pipes(Z,pipe_radius_outer,pipe_radius_inner,dt,num_steps,K)

    # Create test data
    T_inlet = 20 + 5 * np.sin(np.linspace(0, 2*np.pi, num_steps))  # Oscillating inlet temperature
    # T_inlet = np.ones(num_steps)*20 #constant
    # v_flow = 2+0.8*np.cos(np.linspace(0, 2*np.pi, num_steps))  # Oscillating flow velocity
    v_flow = np.ones(num_steps)*2 

    # Run simulation
    pipe.simulate_pipe_temperature(T_inlet, v_flow, num_steps)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.title("Water temperature")
    plt.plot(T_inlet, label='Inlet Temperature')
    plt.plot(pipe.T_lossless_out, label = "lossless out")
    plt.plot(pipe.T_pipe_out, label='Pipe temperature')
    plt.plot(pipe.T_real_out, label = 'Real temperature')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Start interactive session
    # import code
    # code.interact(local=locals())
    
