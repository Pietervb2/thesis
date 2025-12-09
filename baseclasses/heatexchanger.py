from node import Node
from pipe import Pipe
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
            - K_hx: pressure loss coefficient [-]
        """
        super().__init__(x,y,z,hex_id)
        self.U = hex_data[0]
        self.As = hex_data[1]
        self.F = hex_data[2]
        self.K_hx = hex_data[3]
        self.Kvs = hex_data[4]
        self.Kv0 = hex_data[5]

        # Consumer connected to the heat exchanger
        self.consumer = consumer

    def initialize_node(self, num_steps, T_init, dt):
        """
        Overriding function in Node Class.
        Initialize the temperature in the node and the consumer parameters.
        """ 

        super().initialize_node(num_steps, T_init, dt)
        self.consumer.initialize_consumer(num_steps, dt)

        self.h = np.zeros(num_steps)
        self.Kv = np.zeros(num_steps)

    
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

        # self.update_valve(N)

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
        mflow_h = pipe.get_mflow(N)

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
    
    def pressure_drop(self):
        """"
        pressure drop over heat exchanger is stated as dp = rho * K_hx * Q^2 
        """
        
        return self.K_hx * 1000 # Assuming water density of 1000 kg/m3
    
    def equal_percentage_valve(self, h):

        return (self.Kvs / self.Kv0)**(h-1) * self.Kvs
    
    def update_valve(self, N):
        """
        Update the valve position based on the consumer outlet temperature using a PI controller.
        """

        # TODO: dont forget to adjust the parameters for Kv to kg/s/sqrt(bar)
        # Initial step 
        if N == 0:
            self.h[N] = 0.0
            self.Kv[N] = self.equal_percentage_valve(self.h[N])
            return

        # # controller part
        # mflow_min = 0.01
        # if self.consumer.mflow[N] > mflow_min:
            
        #     # implement some PI controller to determine the valve lift
        #     P_k = 
        #     I_k = 
        #     T_set_point = 55 # []
        #     h = P_k * (T_set_point - self.consumer.Tc_out[N]) + I_k * ...

        #     Kv = self.equal_percentage_valve(h)

        #     self.h[N] = h
        #     self.Kv[N] = Kv

        # else:

            # no change in valve position
            self.h[N] = self.h[N-1]
            self.Kv[N] = self.Kv[N-1]
            



