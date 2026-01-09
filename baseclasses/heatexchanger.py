from node import Node

import numpy as np

class HeatExchanger(Node):

    def __init__(self,
                 x,
                 y,
                 z,
                 hex_id: str,
                 hex_data: list,
                 h_initial: float,
                 consumer: object):
        """"
        Args:
            - U: Overall heat transmission coefficient [W / m2 K]
            - As: total transfer area [m2]
            - F: correction factor [-]
            - Kp_rho_dp: pressure loss coefficient [-]
            - Kvs: hydraulic conductivity of fully open valve [m3/h/bar^0.5]
            - Kv0: hydraulic conductivity of closed valve [m3/h/bar^0.5]
            - h_initial: initial valve position (0-1) [-]
            - Kp: Proportional gain of the PI controller [-]
            - Ki: Integral gain of the PI controller [-]

        """
        super().__init__(x,y,z,hex_id)
        self.U = hex_data[0]
        self.As = hex_data[1]
        self.Kp_rho_dp = hex_data[2] * 1000 # Assuming water density of 1000 kg/m3
        self.Kvs = hex_data[3]
        self.Kp = hex_data[4] 
        self.Ki = hex_data[5] 
        self.h_initial = h_initial # Initial position of valve

        # Consumer connected to the heat exchanger
        self.consumer = consumer

    def initialize_node(self, num_steps, T_init, dt):
        """
        Overriding function in Node Class.
        Initialize the temperature in the node and the consumer parameters.
        """ 

        super().initialize_node(num_steps, T_init, dt)
        self.consumer.initialize_consumer(num_steps, dt)

        self.h = np.ones(num_steps)*self.h_initial
        self.Kv = np.ones(num_steps)*self.equal_percentage_valve(self.h_initial)

        self.dt = dt # For PI controller
        self.I_array = np.zeros(num_steps)
        self.I = 0  # Integral term for PI controller

    
    def set_T(self, N):
        """
        Overriding function in Node Class. 
        Sets the temperature of the node to the output temperature of the heat exchanger
        Updates also the consumer outlet temperature.
        """         

        Tc_out, Th_out = HeatExchanger.NTU_method(self,N)
        
        for _, pipe in self.pipes_in.items():
            
            self.T[N] = Th_out

            # Set temperature per pipe
            for _, pipe in self.pipes_out.items():
                pipe.set_T_in(self.T[N], N)

        self.update_valve(N)

    def NTU_method(self,N):

        """
        Calculates the outlet temperatures of the heat exchanger using the NTU method.

        Args:
            N: current time step

        Returns:
            Tc_out: Cold side outlet temperature [C]
            Th_out: Hot side outlet temperature [C]
        """

        # Get inlet temperatures and mass flow rates
        pipe = self.pipes_in[f'Pipe {self.node_id.split()[-1]}.3'] #Assuming single inlet pipe
        Th_in = pipe.T[N]
        mflow_h = pipe.get_mflow(N)

        Tc_in = self.consumer.Tc_in[N]
        mflow_c = self.consumer.mflow[N]

        if mflow_h < 1e-6 or mflow_c < 1e-6:
            # If either mass flow rate is zero, no heat exchange occurs
            self.consumer.Tc_out[N] = Tc_in
            self.consumer.Q_supply[N] = 0
            return Tc_in, Th_in

        # Heat capacity rates
        Cc = mflow_c * pipe.c_water
        Ch = mflow_h * pipe.c_water         

        Cmin = min(Cc, Ch)
        Cmax = max(Cc, Ch)
        Cr = Cmin / Cmax

        NTU = (self.U * self.As) / Cmin

        # Effectiveness calculation for counterflow heat exchanger
        if Cr != 1:
            epsilon = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        else:
            epsilon = NTU / (1 + NTU)

        Q = epsilon * Cmin * (Th_in - Tc_in)

        Tc_out = Tc_in + Q / Cc
        Th_out = Th_in - Q / Ch

        self.consumer.Tc_out[N] = Tc_out
        self.consumer.Q_supply[N] = (Tc_out - Tc_in) * mflow_c * pipe.c_water

        return Tc_out, Th_out
      
    def equal_percentage_valve(self, h):
        """
        Returns Kv value of the valve following the formula for an equal percentage valve, based on the valve displacement.
        For valve displacement lower than h_star Kv does not follow the standard form of the function. It becomes unpredictable. For this region we simply assume a linear behavior.
        """        
        Kv0 = self.Kvs/50
        Kvleak = self.Kvs/2000
        h_star = 0.05

        if h < h_star:
            Kv_star = (self.Kvs/Kv0) ** (h_star-1) * self.Kvs
            Kv = Kvleak + h*(Kv_star - Kvleak)/h_star
        else:
            Kv = (self.Kvs/Kv0) ** (h-1) * self.Kvs
        return Kv
    
    def update_valve(self, N):
        """
        Update the valve position based on the consumer outlet temperature using a PI controller.
        """
        if N < len(self.h)-1:
            if self.consumer.mflow[N] > 0:
                
                # implement PI controller to determine the valve lift

                T_set_point = 60 # Temperature set point for the tapwater outlet [C]
                dT = (T_set_point - self.consumer.Tc_out[N])
                
                self.I += dT * self.dt
                h = self.Kp * dT + self.Ki * self.I
                
                h = min(1,max(0,h)) # As h is scaled to 0-1

                # Anti-windup 
                if (h == 0 and dT < 0) or (h == 1 and dT > 0):
                    self.I -= dT * self.dt  # unwind integral

                Kv = self.equal_percentage_valve(h)

                self.h[N+1] = h
                self.Kv[N+1] = Kv
                self.I_array[N] = self.I
            else:
                # no change in valve position
                self.h[N+1] = 0
                self.Kv[N+1] = self.equal_percentage_valve(0)
                



