from pipe import Pipe
import numpy as np

class Pump(Pipe):

    def __init__(self,
                 pump_id: str,
                 L: float,
                 delta_z: float,
                 pipe_data: list[float],
                 dp_pump: float):
        """
        Args:
            - dp_pump: Pressure increase provided by the pump [Pa]
        """
        super().__init__(pump_id, L, delta_z, pipe_data)
        self.dp_pump = dp_pump

    def pressure_head(self):
        """
        Calculate pressure head loss (negative for pump) provided by the pump
        """

        return self.dp_pump
    
    #NOTE: now within the pipe of the pump the pressure head due to elevation and friciton loss are also calculated. 
    # And the node method is performed in the pipe. But in set_T_network_rec it is made sure that the temperature at the first node is not changed.