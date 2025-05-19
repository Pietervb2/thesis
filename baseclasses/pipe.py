import numpy as np
from typing import Union

class Pipe:

    def __init__(
            self, 
            pipe_id : str, 
            pipe_length : float, 
            pipe_radius_outer : float,
            pipe_radius_inner : float,
            K : float,
            cp_pipe_mat : float,
            rho_pipe_mat : float,
            cp_insu : float,
            rho_insu : float,
            insu_thickness : float
            ):
        """
        Initialize pipe.

        Args: 
        pipe_length [m],
        pipe_radius_outer [m],
        pipe_radius_inner [m],
        K: overall heat transmission coefficient [W/m2K]

        #TODO: update
        """
        self.pipe_id = pipe_id 

        self.L = pipe_length
        self.radius_outer = pipe_radius_outer 
        self.radius_inner = pipe_radius_inner
        self.K = K
        self.insu_thickness = insu_thickness

        # Physical constants       
        self.rho_water = 1e3 # [kg/m3] NOTE: maybe later make a constant file. When there are too many constants.
        self.c_water = 4.18e3 # [J/kg K] specific heat capacity

        self.rho_pipe_mat = rho_pipe_mat #  [kg/m3]
        self.cp_pipe_mat = cp_pipe_mat # [J/kg K] specific heat capacity of steel
        self.Cp_pipe_mat = np.pi * (self.radius_outer ** 2 - self.radius_inner ** 2) * self.rho_pipe_mat * self.cp_pipe_mat * self.L # [J/K] total heat capacity pipe

        self.rho_insu = rho_insu
        self.cp_insu = cp_insu 
        self.Cp_insu = np.pi * ((self.radius_outer + self.insu_thickness) ** 2 - self.radius_outer ** 2) * self.rho_insu * self.cp_insu * self.L # [J/K] total heat capacity insulation

        self.Cp_whole_pipe = self.Cp_pipe_mat + self.Cp_insu
        # self.Cp_whole_pipe = self.Cp_pipe_mat
        self.inner_cs = np.pi * self.radius_inner ** 2 # inner cross section area
        self.outer_cs = np.pi * self.radius_outer ** 2 # outer cross section area


    def bnode_init(self, 
            dt : float,
            num_steps : int,
            v_flow_array : np.ndarray[Union[float]],
            T_inlet_array : np.ndarray[Union[float]],
            T_init: float):
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
        min_steps = int(np.ceil((self.L + v_flow_array[0] * self.dt) / (v_flow_array[0] * self.dt)))
        self.hist_len = min_steps + 5 # Add some margin to ensure we have enough history NOTE: based on what?
        
        # Initialize velocity and temperature of water 
        self.v_history = np.ones(self.hist_len) * v_flow_array[0]
        self.T_history = np.ones(self.hist_len) * T_init
        # self.T_history = np.ones(self.hist_len) * 20 #### HARD CODED NEED TO CHANGE debug

        # Voor zodadelijk uitwerken! Kijken of ik het in de pipe class allemaal moet op slaan of het per keer moet berekenen. 
        self.m_flow_extended = np.concatenate([self.v_history, v_flow_array]) * self.inner_cs * self.rho_water
        self.T_in_extended = np.concatenate([self.T_history, T_inlet_array])

        # initialize temperature arrays
        self.T_lossless = np.ones(self.num_steps) # water temperature at the pipe output without heat loss or capacity of the pipe # debug
        self.T_cap = np.ones(self.num_steps) # water temperature at the pipe output with heat loss  # debug
        self.T = np.ones(self.num_steps)  # real water temperature at the pipe output

        # Initialize flow array without history
        self.m_flow = np.ones(self.num_steps)

        self.T_pipe = np.ones(self.num_steps) * T_init  # temperature of the pipe NOTE: maybe use a different initialization, but for now it is good. It is longer than it should be but it works better with actual_N

        self.t_stay_array = np.ones(self.num_steps) # debug 

    def bnode_method(self,
                     T_ambt : float,
                     N : int):
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

        self.m_flow[N] = self.m_flow_extended[N_hist]


        # TODO: needs to be done more elegantly
        m_flow_ex = self.m_flow_extended
        T_k = self.T_in_extended
        
        self.T_ambt = T_ambt 

        # Find smallest integer n and corresponding R
        n = 0
        while True:
            R = sum(m_flow_ex[N_hist-n:N_hist+1] * self.dt)
            if R > self.L * self.inner_cs * self.rho_water:
                break
            n += 1
        
        # Find smallest integer m
        m = 0
        while True:
            sum_term = sum(m_flow_ex[N_hist-m:N_hist+1] * self.dt)
            if sum_term > self.L * self.inner_cs * self.rho_water + m_flow_ex[N_hist] * self.dt:
                break
            m += 1
        
        # Compute Y and S
        Y = sum(m_flow_ex[N_hist-m+1:N_hist-n] * T_k[N_hist-m+1:N_hist-n] * self.dt)
        
        # Calculate S based on conditions
        if m > n:
            S = sum(m_flow_ex[N_hist-m+1:N_hist+1] * self.dt)
        else:  # m == n
            S = R
        
        # Compute the lossless output temperature T_N
        self.T_lossless[N] = ((R - self.L * self.inner_cs * self.rho_water) * T_k[N_hist-n] + Y + (m_flow_ex[N_hist] * self.dt - S + self.L * self.inner_cs * self.rho_water) * T_k[N_hist-m]) / (m_flow_ex[N_hist] * self.dt)
        
        # Take into account the heat capacity of the steel pipe
        if N - 1 < 0: # for beginning pipe temperature
            self.T_cap[N] = (m_flow_ex[N_hist] * self.c_water * self.T_lossless[N] * self.dt + self.Cp_whole_pipe * self.T_pipe[N]) / (self.Cp_whole_pipe + m_flow_ex[N_hist] * self.c_water * self.dt)
        else:
            self.T_cap[N] = (m_flow_ex[N_hist] * self.c_water * self.T_lossless[N] * self.dt + self.Cp_whole_pipe * self.T_pipe[N-1]) / (self.Cp_whole_pipe + m_flow_ex[N_hist] * self.c_water * self.dt)

        self.T_pipe[N] = self.T_cap[N] # update temperature pipe

        # determine average delay in the pipe
        t_stay = self.average_delay_bnode(n,m,R,S,m_flow_ex,N_hist)

        self.t_stay_array[N] = t_stay

        # print(f't_stay {t_stay}') #debug 

        # self.T[N] = self.T_ambt + (self.T_cap[N] - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section   
        self.T[N] = self.T_ambt + (self.T_lossless[N] - self.T_ambt) * np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section   

        # print(f' temperature {self.T[N]}, {self.pipe_id}') # debug
       

    def average_delay_bnode(self,n,m,R,S,m_flow_ex,N):
        """    
        Source: Maurer. J, Comparison of discrete dynamic pipeline models for operational optimization of District Heating Networks. 2021

        I use the same type of analogy to calculate the mass outflow from the pipe at timestep N.

        Args:
        n : The number of complete pipe lengths traversed.
        m : The number of time steps for the final partial pipe length.
        N : Total number of time steps.
        R : The remaining distance at the start of the calculation.
        S : The remaining distance at the end of the calculation.
        m_flow_ex : Array of mass flows at in let of the pipe
        
        Returns:
        The average delay time of the water in the pipe.
        """

        first_term_delay = n * (R - self.L * self.inner_cs  * self.rho_water)
        second_term_delay = 0
        for i in range(N-m+1, N-n):
            second_term_delay += (N - i) * m_flow_ex[i] * self.dt
            
        third_term_delay = m * (m_flow_ex[N] * self.dt + (self.L * self.inner_cs * self.rho_water - S))

        # print(f' n = {n} and m = {m}') # debug
        delay = (first_term_delay + second_term_delay + third_term_delay)/m_flow_ex[N]

        return delay
    
    def set_m_flow_v(self, v_inflow, N):
        """
        Set the mass flow of the pipe based on the inetlet velocity
        """

        self.m_flow_extended[N + self.hist_len] = v_inflow * self.inner_cs * self.rho_water

    def set_m_flow_m(self, m_flow, N):
        """
        Set the mass flow of the pipe basedon the mass flow
        """
 
        self.m_flow_extended[N + self.hist_len] = m_flow
    
    def set_T_in(self, T_inlet, N):
        """
        Set inlet temperature of the pipe
        """ 
        self.T_in_extended[N + self.hist_len] = T_inlet

    def get_m_flow(self, N):
        """ 
        Get the inlet mass flow at timestep N
        """
        return self.m_flow_extended[N + self.hist_len]
    
    def set_m_flow(self, m_flow, N):
        """
        Set the inlet mass flow at timestep N
        """
        
        self.m_flow_extended[N + self.hist_len] = m_flow
    
    def thermal_transmission_coef(self):
        """
        Calculate the thermal transmission coefficient of the pipe.
        TODO: Implement this function
        """
        pass