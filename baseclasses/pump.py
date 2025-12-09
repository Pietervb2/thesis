from pipe import Pipe
import numpy as np

class Pump(Pipe):

    def __init__(self,
                 pump_id: str,
                 L: float,
                 delta_z: float,
                 pipe_data: list[float],
                 pump_data: list[float]):
        """
        Args:
            - dp_pump: Pressure increase provided by the pump [Pa]
        """
        super().__init__(pump_id, L, delta_z, pipe_data)
        self.a = pump_data[0]
        self.b = pump_data[1]
        self.c = pump_data[2]

    def pressure_head(self):
        """
        Calculate pressure head loss (negative for pump) provided by the pump
        """

        return self.dp_pump

    def pump_curve(self, mflow: float) -> float:
        
        a,b,c = [ -4221.7159851,  -18568.07669405,  50000.] # Example pump curve coefficients
        return a * mflow**2 + b * mflow + c
     
    #NOTE: now within the pipe of the pump the pressure head due to elevation and friciton loss are also calculated. 
    # And the node method is performed in the pipe. But in set_T_network_rec it is made sure that the temperature at the first node is not changed.