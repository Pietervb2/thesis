from .node import Node

import numpy as np

class HeatExchanger(Node):

    def __init__(self,
                 x,
                 y,
                 z,
                 hex_id: str,
                 hex_data: list,
                 consumer: object):
        """"
        Args:
            - U: Overall heat transmission coefficient [W / m2 K]
            - As: total transfer area [m2]
            - F: correction factor [-]
            - Kp_rho_dp: pressure loss coefficient [-]
            - Kvs: hydraulic conductivity of fully open valve [m3/h/bar^0.5]
            - Kv0: hydraulic conductivity of closed valve [m3/h/bar^0.5]
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

        # Consumer connected to the heat exchanger
        self.consumer = consumer

    def initialize_node(self, num_steps, T_init, dt):
        """
        Overriding function in Node Class.
        Initialize the temperature in the node and the consumer parameters.
        """ 

        super().initialize_node(num_steps, T_init, dt)
        self.consumer.initialize_consumer(num_steps, dt)
   
    def set_T(self, N):
        """
        Overriding function in Node Class. 
        Sets the temperature of the node to the output temperature of the heat exchanger
        Updates also the consumer outlet temperature.
        """         

        Tc_out, Th_out = HeatExchanger.NTU_method(self,N)
            
        self.T[N] = Th_out
 
        # Set temperature per pipe
        for _, pipe_in in self.pipes_out.items():
            pipe_in.set_T_in(self.T[N], N)

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
        pipe = self.pipes_in[f'Pipe {self.node_id.split()[-1]}.3'] 
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

        UAs = self.compute_UA(mflow_h, mflow_c)

        NTU = (UAs) / Cmin

        # Effectiveness calculation for counterflow heat exchanger
        if Cr != 1:
            epsilon = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        else:
            epsilon = NTU / (1 + NTU)

        Q = epsilon * Cmin * (Th_in - Tc_in)

        Tc_out = Tc_in + Q / Cc
        Th_out = Th_in - Q / Ch

        Q_supply = (Tc_out - Tc_in) * mflow_c * pipe.c_water
        self.consumer.Tc_out[N] = Tc_out
        self.consumer.Q_supply[N] = Q_supply

        return Tc_out, Th_out
      
    
    def compute_UA(self, mflow_h, mflow_c,  n=0.8):
        """
        Compute off-design UA by scaling each side's convective resistance
        with (mdot/mdot_des)^-0.8, then combining in series.
        
        Derivation: h ~ Re^0.8 ~ mdot^0.8 (Dittus-Boelter)
                    R = 1/UA_des * (mdot/mdot_des)^-0.8
                    1/UA = R1 + R2
        """
        mflow_h = max(mflow_h, 1e-6)
        mflow_c = max(mflow_c, 1e-6)

        UA_des = self.U * self.As # design value
        UA1_des = UA2_des = 2 * UA_des # splitting the value over both convection layers

        mflow_h_des = 0.247 # 31e3 / (30 * 4181) kg /s 
        mflow_c_des = 0.15 # 9 l/min, design variable


        R1 = (1 / UA1_des) * (mflow_h / mflow_h_des) ** (-n)
        R2 = (1 / UA2_des) * (mflow_c / mflow_c_des) ** (-n)

        return 1 / (R1 + R2)