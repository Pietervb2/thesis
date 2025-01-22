import numpy as np
from typing import Union

class Pipe:

    def __init__(
            self, 
            pipe_length : float, 
            pipe_radius_outer : float,
            pipe_radius_inner : float,
            K : float):
        """
        Initialize pipe.

        Args: 
        pipe_length [m],
        pipe_radius_outer [m],
        pipe_radius_inner [m],
        K: overall heat transmission coefficient [W/m2K]

        #TODO: update
        """

        self.L = pipe_length
        self.radius_outer = pipe_radius_outer 
        self.radius_inner = pipe_radius_inner
        self.K = K

        # To be later filled in by the node method
        self.m_flow = 0 # mass flow rate [kg/s]

        # Physical constants       
        self.rho_water = 1e3 # [kg/m3] NOTE: maybe later make a constant file. When there are too many constants.
        self.rho_pipe = 8e3 #  [kg/m3] 
        self.c_water = 4.18e3 # [J/kg K] specific heat capacity
        self.c_pipe = 0.466e3 # [J/kg K] specific heat capacity of steel NOTE: DUMMY checked on wiki, needs to be more accurate 
        
        self.C_pipe = np.pi * (self.radius_outer ** 2 - self.radius_inner ** 2) * self.rho_pipe * self.c_pipe * self.L # [J/K] total heat capacity pipe
        self.inner_cs = np.pi * self.radius_inner ** 2 # inner cross section area
        self.outer_cs = np.pi * self.radius_outer ** 2 # outer cross section area

    def bnode_init(self, 
            dt : float,
            num_steps : int,
            T_ambt : float,
            v_init, T_init):
        """
        Initialize history of velocities and temperatures to ensure valid solutions.
        
        This should only be necessary for the initial use of the node method. As later it should remember its previous temperatures. 
        
        Args:
        dt : time step [s]
        num_steps : number of steps the simulation takes
        T_ambt : ambient temperature [C]
        v_init: initial velocity [m/s]
        T_init: initial temperature [C]
        
        Returns:
        v_history: array of historical velocities
        T_history: array of historical temperatures
        """
        
        self.dt = dt
        self.num_steps = num_steps
        self.T_ambt = T_ambt 

        # Calculate minimum history length needed based on pipe length and flow velocity
        min_steps = int(np.ceil((self.L + v_init * self.dt) / (v_init * self.dt)))
        # Add some margin to ensure we have enough history NOTE: based on what?
        history_length = min_steps + 5
        
        # Initialize velocity and temperature of water 
        self.v_history = np.ones(history_length) * v_init
        self.T_history = np.ones(history_length) * T_init

        # initialize temperature for the pipe
        self.T_pipe = np.ones(history_length + self.num_steps) * T_init  # NOTE: maybe use a different initialization, but for now it is good. It is longer than it should be but it works better with actual_N

        self.t_stay_array = np.ones(self.num_steps) # debug 

    def bnode_method(self,
                     v: np.ndarray[Union[float]], 
                     T_k : np.ndarray[Union[float]], 
                     N : int):
        """
        Implementation of b-node method for temperature dynamics in pipes
        based on Benonysson (1991)
        
        Args:
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
        
        # Compute the lossless output temperature T_N
        T_N_lossless = ((R - self.L) * T_k[N-n] + Y + (v[N] * self.dt - S + self.L) * T_k[N-m]) / (v[N] * self.dt)
        
        # Take into account the heat capacity of the steel pipe
        self.m_flow = v[N] * self.rho_water * self.inner_cs  # mass flow 
        
        T_N_pipe = (self.m_flow * self.c_water * T_N_lossless * self.dt + self.C_pipe * self.T_pipe[N-1]) / (self.C_pipe + self.m_flow * self.c_water * self.dt)

        self.T_pipe[N] = T_N_pipe # update temperature pipe

        # determine average delay in the pipe
        t_stay = self.average_delay_bnode(n,m,N,R,S,v)
        self.t_stay_array[N-len(self.v_history)] = t_stay
        print(f't_stay {t_stay}') #debug 

        T = self.T_ambt + (T_N_pipe - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section   
       
        return T_N_lossless, T_N_pipe, T #debug 

    def average_delay_bnode(self,n,m,N,R,S,v):
        """    
        Source: Maurer. J, Comparison of discrete dynamic pipeline models for operational optimization of District Heating Networks. 2021

        Args:
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

    def thermal_transmission_coef(self):
        """
        Calculate the thermal transmission coefficient of the pipe.
        TODO: Implement this function
        """
        pass