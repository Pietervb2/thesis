import numpy as np
import matplotlib.pyplot as plt

class Consumer:

    def __init__(self, 
                 consumer_id: str,
                 A1: float,
                 A2: float,
                 Period1: float,
                 Period2: float,
                 phi1: float,
                 phi2: float, 
                 offset: float,
                 tau: float):
        
        """
        Args:
            - consumer_id: unique id of the consumer [-]
            - A1: Amplitude of first sinusoidal component [W]
            - A2: Amplitude of second sinusoidal component [W]
            - Period1: Period of first sinusoidal component [s]
            - Period2: Period of second sinusoidal component [s]
            - phi1: Phase shift of first sinusoidal component [rad]
            - phi2: Phase shift of second sinusoidal component [rad]
            - offset: Offset of the sinusoidal components [W]
            - tau: Time constant of the consumer [s]

        The periods of the sinusoidal components are fitted to 24h, based on the demand curve used in https://ieeexplore.ieee.org/document/10981678. 
        That's why they are multplied by 3600 to convert hours to seconds in initialize_consumer.
        """
        self.consumer_id = consumer_id
        self.A1 = A1
        self.A2 = A2
        self.Period1 = Period1
        self.Period2 = Period2  
        self.phi1 = phi1
        self.phi2 = phi2
        self.offset = offset
        self.tau = tau

    def initialize_consumer(self, num_steps: int, dt: float) -> None:
        """
        Initialize the consumer parameters.
        Determine the heat demand profile and the mass flow rate over the simulation time.

        Args:
            - num_steps: number of time steps in the simulation [-]
        """
        
        self.Tc_in = np.ones(num_steps)*35 # Cold side inlet temperature [C]. #TODO: for now it is constant. Need to check the constant value. 
        self.Tc_out = np.ones(num_steps) # Cold side outlet temperature [C]
        self.Q_supply = np.zeros(num_steps)  # Supplied heat [W]

        time = np.arange(0,num_steps*dt,dt)

        self.Q_d = 1e3*(self.A1 * np.sin(2 * np.pi / (self.Period1*3600) * (time-self.tau) + self.phi1) +\
                         self.A2 * np.sin(2 * np.pi / (self.Period2*3600) * (time-self.tau) + self.phi2) + self.offset)  # Heat demand [W]

        if self.Q_d[self.Q_d < 0] != 0:
            raise ValueError("Heat demand cannot be negative. Please check the parameters.")
        
        # Pre-calculate mass flow rates based on heat demand. 
        # Assuming 20 K temperature difference over HEX on consumer side.

        delta_T = 20  # Temperature difference [K]
        c_p = 4186    # Specific heat capacity of water [J/(kg·K)]

        self.mflow = self.Q_d / (c_p * delta_T) 

    def __repr__(self):
        return f"Consumer(consumer_id={self.consumer_id}, A1={self.A1}, A2={self.A2}, Period1={self.Period1}, Period2={self.Period2}, phi1={self.phi1}, phi2={self.phi2})"

               
if __name__ == "__main__":
    consumer = Consumer("Consumer_1", 1000, 500, 3600, 7200, 0, np.pi/4, 2)
    time = 1800  # Example time in seconds
    heat_demand = consumer.get_heat_demand(time)
    print(consumer)