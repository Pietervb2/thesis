import numpy as np
import matplotlib.pyplot as plt

class Consumer:

    def __init__(self, 
                 consumer_id: str,
                 demand_type: int,
                 start_time: float,
                   ):
        
        """
        Args:
            - consumer_id: Unique identifier for the consumer [-]
            - demand_type: Type of heat demand profile 
            - start_time: Time when the consumer starts demanding heat [s]

        """
        self.consumer_id = consumer_id
        self.demand_type = demand_type
        self.start_time = start_time

    def initialize_consumer(self, num_steps: int, dt: float) -> None:
        """
        Initialize the consumer parameters.
        Determine the heat demand profile and the mass flow rate over the simulation time.

        Args:
            - num_steps: number of time steps in the simulation [-]
        """
        
        self.Tc_in = np.ones(num_steps)*10 # Cold side inlet temperature [C]. #TODO: for now it is constant. Need to check the constant value. 
        self.Tc_out = np.zeros(num_steps) # Cold side outlet temperature [C]
        self.Q_supply = np.zeros(num_steps)  # Supplied heat [W]

        self.generate_heat_demand(num_steps,dt)

        if self.Q_d[self.Q_d < 0] != 0:
            raise ValueError("Heat demand cannot be negative. Please check the parameters.")
        
        # Pre-calculate mass flow rates based on heat demand. Only for tap water. 
        # Which is always 10 degrees, and we always want 60 degrees so dT = 50

        delta_T = 50  # Temperature difference [K]
        c_p = 4186    # Specific heat capacity of water [J/(kg·K)]

        self.mflow = self.Q_d / (c_p * delta_T) 

        self.mflow[self.mflow < 1e-2] = 0  # Set very small mass flow rates to zero

    def generate_heat_demand(self, num_steps, dt):

        #NOTE: later extend to the posibility of multiple showers in a day. 

        t = np.arange(0,num_steps*dt,dt)
        self.Q_d = np.zeros(num_steps)

        for i in range(len(self.demand_type)):

            demand_type = self.demand_type[i]
            start_time = self.start_time[i]

            if demand_type == 'shower':

                # Time intervals
                t1_start, t1_end = start_time*3600, (start_time + 0.05)*3600
                t2_start, t2_end = t1_end, (start_time + 0.1)*3600
                t3_start, t3_end = t2_end, (start_time + 0.164)*3600

                # Heat demand height
                h1, h2, h3 = 20e3, 25e3, 30e3

                Q_d = np.zeros_like(t)

                # Drie blockgolven naast elkaar
                Q_d[(t >= t1_start) & (t < t1_end)] = h1
                Q_d[(t >= t2_start) & (t < t2_end)] = h2
                Q_d[(t >= t3_start) & (t < t3_end)] = h3

                self.Q_d += Q_d
            
            elif demand_type == 'constant':

                Q_d = np.ones(num_steps)*15e3
                self.Q_d += Q_d

            elif demand_type == "nothing":
                Q_d = np.zeros(num_steps)
                self.Q_d += Q_d

            else:
                raise KeyError(f"Wrong heat demand input type for {self.consumer_id}, key was {demand_type}")

    def set_Q_supply(self, Q_supply, N):   
        self.Q_supply[N] = Q_supply

    def set_Tc_out(self, Tc_out, N):
        self.Tc_out[N] = Tc_out

    def __repr__(self):
        return f"Consumer(consumer_id={self.consumer_id}, A1={self.A1}, A2={self.A2}, Period1={self.Period1}, Period2={self.Period2}, phi1={self.phi1}, phi2={self.phi2})"

               
if __name__ == "__main__":
    pass