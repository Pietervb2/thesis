import numpy as np
from typing import Union

class Pipe:

    def __init__(
            self, 
            pipe_id : str, 
            pipe_length : float, 
            delta_z : float,
            pipe_data : list[float],
            hex_pipe : bool = False,
            ):
        """
        Initialize pipe.

        Args: 
        pipe_id : unique identifier for the pipe
        pipe_length : length of the pipe [m]
        delta_z : elevation difference of the pipe [m]
        pipe_data : list of physical properties of the pipe and insulation
        hex_pipe : boolean indicating if the pipe is connected to a heat exchanger
        """
        self.pipe_id = pipe_id 
        self.L = pipe_length
        self.delta_z = delta_z
        self.hex_pipe = hex_pipe

        self.r_inner = pipe_data[0]
        self.r_outer = pipe_data[1]
        self.rho_pipe = pipe_data[2]
        self.rho_insu = pipe_data[3]
        self.cp_pipe = pipe_data[4]
        self.cp_insu = pipe_data[5]

        self.insu_thickness = pipe_data[6]
        self.K = pipe_data[7]
        self.epsilon = pipe_data[8]
        # self.Re = pipe_data[9]
       
        # Physical constants       
        self.rho_water = 1e3 # [kg/m3] 
        self.c_water = 4.186e3 # [J/kg K] specific heat capacity
       
        self.inner_cs = np.pi * self.r_inner ** 2 # inner cross section area
        self.outer_cs = np.pi * self.r_outer ** 2 # outer cross section area
        
        self.C_pipe = (self.outer_cs - self.inner_cs) * self.rho_pipe * self.cp_pipe * self.L # [J/K] total heat capacity pipe      
        self.C_insu = np.pi * ((self.r_outer + self.insu_thickness) ** 2 - self.r_outer ** 2) * self.rho_insu * self.cp_insu * self.L # [J/K] total heat capacity insulation
        self.C_whole_pipe = self.C_pipe + self.C_insu

    def bnode_init(self, 
            dt : float,
            num_steps : int,
            T_init_water: float,
            T_init_pipe: float,
            v_inflow: float
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

        #TODO: update
        """     
        self.dt = dt
        self.num_steps = num_steps

        # Calculate minimum history length needed based on pipe length and flow velocity
        min_steps = int(np.ceil((self.L + v_inflow * self.dt) / (v_inflow * self.dt)))
        self.hist_len = min_steps + 5 # Add some margin to ensure we have enough history NOTE: based on what?
        
        # Initialize history velocity and temperature of water 
        self.v_history = np.ones(self.hist_len) * v_inflow 
        self.T_history = np.ones(self.hist_len) * T_init_water

        self.mflow_extended = np.round(np.concatenate([self.v_history, np.zeros(num_steps)]) * self.inner_cs * self.rho_water,5)
        self.T_in_extended = np.round(np.concatenate([self.T_history, np.zeros(num_steps)]),5)

        # T_lossless: water temperature at the pipe output without heat loss or capacity of the pipe 
        # T_cap: water temperature at the pipe output with heat loss to capacity of the pipe  
        # T: real water temperature at the pipe output
        self.T_lossless = np.ones(self.num_steps) * T_init_water
        self.T_cap = np.ones(self.num_steps) * T_init_water
        self.T = np.ones(self.num_steps) * T_init_water
        
        # Temperature of the pipe 
        self.T_pipe = np.ones(self.num_steps) * T_init_pipe  

        # Initialize flow array without history to save the eventual flow and temperature in the pipe
        self.mflow = np.zeros(self.num_steps)
        self.dp_friction_array = np.zeros(self.num_steps)

        # Save average time delay in pipe
        self.t_stay_array = np.ones(self.num_steps) 

        # Try to make model faster
        self.cum_mass = np.zeros_like(self.mflow_extended)
        self.cum_mT   = np.zeros_like(self.mflow_extended)

        self.cum_mass[:self.hist_len+1] = np.cumsum(self.mflow_extended[:self.hist_len+1] * self.dt)
        self.cum_mT[:self.hist_len+1]   = np.cumsum(self.mflow_extended[:self.hist_len+1] * self.T_in_extended[:self.hist_len+1] * self.dt)
   
    def bnode_method_fast(self,
                 T_ambt: float,
                 N: int,
                 no_cap=False):
        """
        Implementation of b-node method for temperature dynamics in pipes
        based on Benonysson (1991), using searchsorted for efficiency.
        
        Args:
        T_ambt: ambient temperature [C]
        N: current time step number
        no_cap : if True, ignore the heat capacity of the pipe (for testing purposes)
        
        Returns:
        T_N: Temperature of water flowing from the pipe at time step N
        """

        N_hist = N + self.hist_len
        mflow_ex, T_in_ex = self.mflow_extended, self.T_in_extended
        self.mflow[N] = mflow_ex[N_hist]

        self.cum_mass[N_hist] = self.cum_mass[N_hist-1] + mflow_ex[N_hist] * self.dt
        self.cum_mT[N_hist]   = self.cum_mT[N_hist-1] + mflow_ex[N_hist] * T_in_ex[N_hist] * self.dt

        if mflow_ex[N_hist] != 0:
            # Replace the while loops with searchsorted for efficiency
            M_pipe = self.L * self.inner_cs * self.rho_water

            target_n = self.cum_mass[N_hist] - M_pipe
            idx_n = np.searchsorted(self.cum_mass[:N_hist+1], target_n, side='left')           
            n = N_hist - idx_n
            R = self.cum_mass[N_hist] - (self.cum_mass[idx_n-1] if idx_n > 0 else 0.0)

            target_m = self.cum_mass[N_hist] - (M_pipe + mflow_ex[N_hist] * self.dt)
            idx_m = np.searchsorted(self.cum_mass[:N_hist+1], target_m, side='left')
            m = N_hist - idx_m

            # print(f"Step {N}: n={n}, R={R:.2f},R_length {R/(self.inner_cs*self.rho_water)} m={m},fast") #debug

            Y = sum(mflow_ex[N_hist-m+1:N_hist-n] * T_in_ex[N_hist-m+1:N_hist-n] * self.dt)
            S = sum(mflow_ex[N_hist - m + 1:N_hist + 1] * self.dt) if m > n else R

            # Compute lossless temperature with weighted average
            lossless = (
                (R - M_pipe) * T_in_ex[N_hist-n]
                + Y
                + (mflow_ex[N_hist] * self.dt - S + M_pipe) * T_in_ex[N_hist-m]
            ) / (mflow_ex[N_hist] * self.dt)
            self.T_lossless[N] = lossless

            # Take into account the heat capacity of the steel pipe
            prev_Tpipe = self.T_pipe[N-1] if N > 0 else self.T_pipe[N]
            self.T_cap[N] = (
                mflow_ex[N_hist] * self.c_water * self.T_lossless[N] * self.dt
                + self.C_whole_pipe * prev_Tpipe
            ) / (self.C_whole_pipe + mflow_ex[N_hist] * self.c_water * self.dt)

            # Update pipe wall temperature
            self.T_pipe[N] = self.T_cap[N]  

            # Average delay in pipe
            t_stay = self.average_delay_bnode(n, m, R, S, mflow_ex, N_hist)
            self.t_stay_array[N] = t_stay

            # Final outlet water temperature including heat loss to ambient
            ref_T = self.T_lossless[N] if no_cap else self.T_cap[N]
            decay = np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.inner_cs))
            self.T[N] = T_ambt + (ref_T - T_ambt) * decay
        else:
            # Incase there is no mass flow the bnode method needs to be treated differently           
            if N == 0:
                # no flow, so output temp is set equal to initial temperature of the water. Which can be found in the input history
                self.T_lossless[N] = self.T_in_extended[N_hist-1] 
                self.T_cap[N] = self.T_in_extended[N_hist-1]  
                self.T[N] = self.T_in_extended[N_hist-1]
            else:
                self.T_lossless[N] = self.T_lossless[N-1]  # no flow, so output temp remains the same as previous time step
                self.T_cap[N] = self.T_cap[N-1]
                
                # Update temperature pipe wall
                self.T_pipe[N] = self.T_cap[N]
                
                # Update time stay in pipe
                t_stay = self.dt + self.t_stay_array[N-1]
                self.t_stay_array[N] = t_stay
                
                # Final outlet water temperature including heat loss to ambient
                ref_T = self.T_lossless[N] if no_cap else self.T_cap[N]

                decay = np.exp(-self.K * t_stay / (self.rho_water * self.c_water * self.inner_cs) ) # NOTE: I used here the inner cross section   
                self.T[N] = T_ambt + (ref_T - T_ambt) * decay   

    def average_delay_bnode(self,n,m,R,S,mflow_ex,N):
        """    
        Source: Maurer. J, Comparison of discrete dynamic pipeline models for operational optimization of District Heating Networks. 2021

        I use the same type of analogy to calculate the mass outflow from the pipe at timestep N.

        Args:
        n : The number of complete pipe lengths traversed.
        m : The number of time steps for the final partial pipe length.
        N : Total number of time steps.
        R : The remaining distance at the start of the calculation.
        S : The remaining distance at the end of the calculation.
        mflow_ex : Array of mass flows at in let of the pipe
        
        Returns:
        The average delay time of the water in the pipe.
        """

        first_term_delay = n * (R - self.L * self.inner_cs  * self.rho_water)
        second_term_delay = 0
        for i in range(N-m+1, N-n):
            second_term_delay += (N - i) * mflow_ex[i] * self.dt
            
        third_term_delay = m * (mflow_ex[N] * self.dt + self.L * self.inner_cs * self.rho_water - S)

        delay = (first_term_delay + second_term_delay + third_term_delay)/mflow_ex[N]

        # print(f'N = {N}, self.hist_len = {self.hist_len}, n = {n}, m = {m}, R = {R:.2f}, S = {S:.2f}, t_stay {delay}, mflow_ex[N] = {mflow_ex[N]}') #debug


        return delay
    
    def pressure_friction(self):
        """
        Darcy Weisbach equation to calculate pressure drop in pipe
        Haaland method to determine frictor factor f

        #TODO: update Re per sim with flow velocity based calculation, but could lead to changing whole equation. 
        """      
        # Pipe pressure loss over HEX already in Kp term of HEX. And setting to 0 gives problem for the jacobian
        if self.hex_pipe:
            return 0.1
        
        D = self.r_inner*2
        Re = 10e3 
        log_term = ((self.epsilon/D)/3.7)**1.11 + (6.9/Re)
        f = (1 / (-1.8 * np.log10(log_term)))**2       
        return 8 * f * self.L / (np.pi ** 2 * D ** 5 * self.rho_water)                                                    

    def pressure_elevation(self):
        """
        Calculate pressure head due to elevation difference. 
        Also included for the pipes connected to the HEX, in contract to the friction pressure term. Because in loop elevation pressure will disappear. 
        """
        return 9.81 * self.rho_water * self.delta_z
    
    def set_T_in(self, T_inlet, N):
        """
        Set inlet temperature of the pipe
        """ 
        self.T_in_extended[N + self.hist_len] = T_inlet

    def get_mflow(self, N):
        """ 
        Get the inlet mass flow at timestep N
        """
        return self.mflow_extended[N + self.hist_len]
    
    def set_mflow(self, mflow, N):
        """
        Set the inlet mass flow at timestep N
        """
        
        self.mflow_extended[N + self.hist_len] = mflow

    def save_dp_friction(self,N):

        """
        Total pressure drop over the pipe due to friction and valves.
        """

        self.dp_friction_array[N] = self.pressure_friction() * self.mflow_extended[N + self.hist_len]**2


if __name__ == "__main__":
    pass