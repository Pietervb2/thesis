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
            v_flow : float,
            T_inlet : float):
        """
        Initialize history of velocities and temperatures to ensure valid solutions.
        
        This should only be necessary for the initial use of the node method. As later it should remember its previous temperatures. 
        
        Args:
        dt : time step [s]
        num_steps : number of steps the simulation takes
        v_init: initial velocity [m/s]
        T_init: initial temperature [C]
        
        Returns:
        v_history: array of historical velocities
        T_history: array of historical temperatures
        """
        
        self.dt = dt
        self.num_steps = num_steps

        # Calculate minimum history length needed based on pipe length and flow velocity
        min_steps = int(np.ceil((self.L + v_flow[0] * self.dt) / (v_flow[0] * self.dt)))
        # Add some margin to ensure we have enough history NOTE: based on what?
        self.hist_len = min_steps + 5
        
        # Initialize velocity and temperature of water 
        self.v_history = np.ones(self.hist_len) * v_flow[0]
        self.T_history = np.ones(self.hist_len) * T_inlet[0]

        # Voor zodadelijk uitwerken! Kijken of ik het in de pipe class allemaal moet op slaan of het per keer moet berekenen. 
        self.v_in_extended = np.concatenate([self.v_history, v_flow])
        self.T_in_extended = np.concatenate([self.T_history, T_inlet])

        # initialize temperature arrays
        self.T_lossless = np.ones(self.num_steps) * T_inlet[0] # water temperature at the pipe output without heat loss or capacity of the pipe # debug
        self.T_cap = np.ones(self.num_steps) * T_inlet[0] # water temperature at the pipe output with heat loss  # debug
        self.T = np.ones(self.num_steps) * T_inlet[0] # real water temperature at the pipe output

        self.T_pipe = np.ones(self.num_steps) * T_inlet[0]  # temperature of the pipe NOTE: maybe use a different initialization, but for now it is good. It is longer than it should be but it works better with actual_N

        self.t_stay_array = np.ones(self.num_steps) # debug 

    def bnode_method(self,
                     N : int,
                     T_ambt : float):
        """
        Implementation of b-node method for temperature dynamics in pipes
        based on Benonysson (1991)
        
        Args:
        v: array of flow velocities at each time step k (including history)
        T_k: array of temperatures at inflow end at each time step k (including history)
        N: current time step number
        T_ambt: ambient temperature [C]
        
        Returns:
        T_N: Temperature of water flowing from the pipe at time step N
        """
        
        # N_hist is for the extended arrays containing the fictive history of the pipe. The normal N is for the actual time step.
        N_hist = N + self.hist_len

        # TODO: needs to be done more elegantly
        v = self.v_in_extended
        T_k = self.T_in_extended
        
        self.T_ambt = T_ambt 

        # Find smallest integer n and corresponding R
        n = 0
        while True:
            R = sum(v[N_hist-n:N_hist+1] * self.dt)
            if R > self.L:
                break
            n += 1
        
        # Find smallest integer m
        m = 0
        while True:
            sum_term = sum(v[N_hist-m:N_hist+1] * self.dt)
            if sum_term > self.L + v[N_hist] * self.dt:
                break
            m += 1
        
        # Compute Y and S
        Y = sum(v[N_hist-m+1:N_hist-n] * T_k[N_hist-m+1:N_hist-n] * self.dt)
        
        # Calculate S based on conditions
        if m > n:
            S = sum(v[N_hist-m+1:N_hist+1] * self.dt)
        else:  # m == n
            S = R
        
        # Compute the lossless output temperature T_N
        self.T_lossless[N] = ((R - self.L) * T_k[N_hist-n] + Y + (v[N_hist] * self.dt - S + self.L) * T_k[N_hist-m]) / (v[N_hist] * self.dt)
        
        # Take into account the heat capacity of the steel pipe
        self.calc_mass_flow(v[N_hist])
        
        self.T_cap[N] = (self.m_flow * self.c_water * self.T_lossless[N] * self.dt + self.C_pipe * self.T_pipe[N-1]) / (self.C_pipe + self.m_flow * self.c_water * self.dt)

        self.T_pipe[N] = self.T_cap[N] # update temperature pipe

        # determine average delay in the pipe
        t_stay = self.average_delay_bnode(n,m,N_hist,R,S,v)
        self.t_stay_array[N-len(self.v_history)] = t_stay

        # print(f't_stay {t_stay}') #debug 

        self.T[N] = self.T_ambt + (self.T_cap[N] - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section   
       

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

        # print(f' n = {n} and m = {m}') # debug

        return (first_term + second_term + third_term)/v[N]
    

    def set_inlet_temperature(self, T_inlet, N):
        """
        Set the inlet temperature of the pipe based on the temperature of the node
        """
        N_hist = N + self.hist_len

        self.T_in_extended[N_hist] = T_inlet
    
    def set_m_flow(self, m_flow, N):
        """
        Set the inlet mass flow of the Nth water plug, coming from the attached node.
        """
        
        N_hist = N + self.hist_len

        self.v_in_extended[N_hist] = m_flow

    def thermal_transmission_coef(self):
        """
        Calculate the thermal transmission coefficient of the pipe.
        TODO: Implement this function
        """
        pass

    def calc_mass_flow(self, v):

        self.m_flow = self.rho_water * self.inner_cs * v