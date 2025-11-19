from node import Node
from pipe import Pipe
import numpy as np

class HeatExchanger(Node):

    def __init__(self,
                 x,
                 y,
                 z,
                 hex_id: str,
                 U: float,
                 As: float,
                 F: float,
                 K_hx: float,
                 consumer: object):
        """"
        Args:
            - U: Overall heat transmission coefficient [W / m2 K]
            - As: total transfer area [m2]
            - F: correction factor [-]
            - K_hx: pressure loss coefficient [-]
        """
        super().__init__(x,y,z,hex_id)
        self.U = U
        self.As = As
        self.F = F
        self.K_hx = K_hx

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
        
        for _, pipe in self.pipes_in.items():
            
            self.T[N] = Th_out

            # Set temperature per pipe
            for _, pipe in self.pipes_out.items():
                pipe.set_T_in(self.T[N], N)

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
        pipe = self.pipes_in[f'Pipe {self.node_id.split()[-1]}.1'] #Assuming single inlet pipe
        Th_in = pipe.T[N]
        mflow_h = pipe.get_m_flow(N)

        Tc_in = self.consumer.Tc_in[N]
        mflow_c = self.consumer.mflow[N]

        if mflow_h < 1e-6 or mflow_c < 1e-6:
            # If either mass flow rate is zero, no heat exchange occurs
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

        Q = self.F * epsilon * Cmin * (Th_in - Tc_in)

        Tc_out = Tc_in + Q / Cc
        Th_out = Th_in - Q / Ch

        self.consumer.Tc_out[N] = Tc_out
        self.consumer.Q_supply[N] = (Tc_out - Tc_in) * mflow_c * pipe.c_water

        return Tc_out, Th_out


