from node import Node
from pipe import Pipe

class HeatExchanger(Node):

    def __init__(self,
                 U,
                 As,
                 F,
                 K_hx):
        """"
        Args:
            - U: Overall heat transmission coefficient [W / m2 K]
            - As: total transfer area [m2]
            - F: correction factor [-]
            - K_hx: pressure loss coefficient [-]
        """
        super().__init__()
        self.U = U
        self.As = As
        self.F = F
        self.K_hx = K_hx

        def set_T(self, N, T_out):
            """
            Overriding function in Node Class. 
            Sets the temperature of the node to the output temperature of the heat exchanger
            """         
            for _, pipe in self.pipes_in.items():
                
                self.T[N] = T_out

                # Set temperature per pipe
                for _, pipe in self.pipes_out.items():
                    pipe.set_T_in(self.T[N], N)


        def NTU_method(self,N):

            
