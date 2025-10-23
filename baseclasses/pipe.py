import numpy as np
from typing import Union

class Pipe:

    def __init__(
            self, 
            pipe_id : str, 
            pipe_length : float, 
            pipe_r_inner : float,
            pipe_r_outer : float,
            cp_pipe : float,
            rho_pipe : float,
            cp_insu : float,
            rho_insu : float,
            insu_thickness : float,
            K : float        
            ):
        """
        Initialize pipe.

        Args: 
        pipe_length [m],
        pipe_r_outer [m],
        pipe_r_inner [m],
        K: overall heat transmission coefficient [W/m2K]

        #TODO: update
        """
        self.pipe_id = pipe_id 

        self.L = pipe_length
        self.r_outer = pipe_r_outer 
        self.r_inner = pipe_r_inner
        self.K = K
        self.insu_thickness = insu_thickness

        # Physical constants       
        self.rho_water = 1e3 # [kg/m3] NOTE: maybe later make a constant file. When there are too many constants.
        self.c_water = 4.18e3 # [J/kg K] specific heat capacity

        self.rho_pipe = rho_pipe #  [kg/m3]
        self.cp_pipe = cp_pipe # [J/kg K] specific heat capacity of steel

        self.inner_cs = np.pi * self.r_inner ** 2 # inner cross section area
        self.outer_cs = np.pi * self.r_outer ** 2 # outer cross section area
        self.C_pipe = (self.outer_cs - self.inner_cs) * self.rho_pipe * self.cp_pipe * self.L # [J/K] total heat capacity pipe

        self.rho_insu = rho_insu
        self.cp_insu = cp_insu 
        self.C_insu = np.pi * ((self.r_outer + self.insu_thickness) ** 2 - self.r_outer ** 2) * self.rho_insu * self.cp_insu * self.L # [J/K] total heat capacity insulation

        self.C_whole_pipe = self.C_pipe + self.C_insu

    def bnode_init(self, 
            dt : float,
            num_steps : int,
            v_flow_array : np.ndarray[Union[float]],
            T_inlet_array : np.ndarray[Union[float]],
            T_init_water: float,
            T_init_pipe: float
            ):
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
        
        # Initialize history velocity and temperature of water 
        self.v_history = np.ones(self.hist_len) * v_flow_array[0]
        self.T_history = np.ones(self.hist_len) * T_init_water

        #NOTE Voor zodadelijk uitwerken! Kijken of ik het in de pipe class allemaal moet op slaan of het per keer moet berekenen. 
        self.m_flow_extended = np.round(np.concatenate([self.v_history, v_flow_array]) * self.inner_cs * self.rho_water,3)
        self.T_in_extended = np.concatenate([self.T_history, T_inlet_array])

        self.T_lossless = np.ones(self.num_steps) * T_init_water
        self.T_cap = np.ones(self.num_steps) * T_init_water
        self.T = np.ones(self.num_steps) * T_init_water
        # T_lossless: water temperature at the pipe output without heat loss or capacity of the pipe 
        # T_cap: water temperature at the pipe output with heat loss  
        # T: real water temperature at the pipe output

        # Initialize flow array without history to save the eventual flow and temperature in the pipe
        self.m_flow = np.ones(self.num_steps)
        self.m_flow[0] = self.m_flow_extended[self.hist_len]
        self.T_pipe = np.ones(self.num_steps) * T_init_pipe  # temperature of the pipe 

        # Save average time delay in pipe
        self.t_stay_array = np.ones(self.num_steps) 

    def bnode_method(self,
                     T_ambt : float,
                     N : int,
                     no_cap = False):
        """
        Implementation of b-node method for temperature dynamics in pipes
        based on Benonysson (1991)
        
        Args:
        T_ambt: ambient temperature [C]
        N: current time step number
        no_cap : if True, ignore the heat capacity of the pipe (for testing purposes)

        
        Returns:
        T_N: Temperature of water flowing from the pipe at time step N
        """
        
        N_hist = N + self.hist_len
        m_flow_ex, T_in_ex = self.m_flow_extended, self.T_in_extended
        self.m_flow[N] = self.m_flow_extended[N_hist]
        
        # Find smallest integer n and corresponding R
        n, R = 0, 0
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
        Y = sum(m_flow_ex[N_hist-m+1:N_hist-n] * T_in_ex[N_hist-m+1:N_hist-n] * self.dt)
        S = sum(m_flow_ex[N_hist - m + 1:N_hist + 1] * self.dt) if m > n else R
         
        # Compute the lossless output temperature T_N
        lossless = (
            (R - self.L * self.inner_cs * self.rho_water) * T_in_ex[N_hist-n]
            + Y
            + (m_flow_ex[N_hist] * self.dt - S + self.L * self.inner_cs * self.rho_water) * T_in_ex[N_hist-m]
            ) / (m_flow_ex[N_hist] * self.dt)
        self.T_lossless[N] = lossless      

        # Take into account the heat capacity of the steel pipe
        # Assume equilibrium in dt between the temperature of the pipe and the water
        prev_Tpipe = self.T_pipe[N-1] if N > 0 else self.T_pipe[N]
        self.T_cap[N] = (
            m_flow_ex[N_hist] * self.c_water * self.T_lossless[N] * self.dt
            + self.C_whole_pipe * prev_Tpipe
            ) / (self.C_whole_pipe + m_flow_ex[N_hist] * self.c_water * self.dt)

        # Update temperature pipe wall
        self.T_pipe[N] = self.T_cap[N] 

        # Determine average delay in the pipe
        t_stay = self.average_delay_bnode(n,m,R,S,m_flow_ex,N_hist)
        self.t_stay_array[N] = t_stay

        # Final outlet water temperature including heat loss to ambient
        ref_T = self.T_lossless[N] if no_cap else self.T_cap[N]

        # t_stay = t_stay / self.dt
        decay = np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.outer_cs) ) # NOTE: I used here the outer cross section   
        self.T[N] = T_ambt + (ref_T - T_ambt) * decay    

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
            
        third_term_delay = m * (m_flow_ex[N] * self.dt + self.L * self.inner_cs * self.rho_water - S)

        # print(f' n = {n} and m = {m}') # debug
        delay = (first_term_delay + second_term_delay + third_term_delay)/m_flow_ex[N]

        return delay
    
    def set_m_flow_v(self, v_inflow, N):
        """
        Set the mass flow of the pipe based on the velocity
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