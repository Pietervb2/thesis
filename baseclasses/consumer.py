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
                 offset):
        
        """
        Args:
            - consumer_id: unique id of the consumer [-]
            - A1: Amplitude of first sinusoidal component [W]
            - A2: Amplitude of second sinusoidal component [W]
            - Period1: Period of first sinusoidal component [s]
            - Period2: Period of second sinusoidal component [s]
            - phi1: Phase shift of first sinusoidal component [rad]
            - phi2: Phase shift of second sinusoidal component [rad]
        """
        self.consumer_id = consumer_id
        self.A1 = A1
        self.A2 = A2
        self.Period1 = Period1
        self.Period2 = Period2  
        self.phi1 = phi1
        self.phi2 = phi2
        self.offset = offset

    def initialize_consumer(self, num_steps: int, dt: float) -> None:
        """
        Initialize the consumer parameters.
        Args:
            - num_steps: number of time steps in the simulation [-]
        """
        
        self.Tc_in = np.ones(num_steps)*40 # Cold side inlet temperature [C]. #TODO: for now it is constant. Need to check the constant value. 
        self.Tc_out = np.ones(num_steps)*40 # Cold side outlet temperature [C]
        self.mflow = np.zeros(num_steps) # Mass flow rate [kg/s]

        times = np.arange(0,num_steps*dt,dt)
        self.Q_d = self.A1 * np.sin(2 * np.pi / (self.Period1*3600) * times + self.phi1) +\
                         self.A2 * np.sin(2 * np.pi / (self.Period2*3600) * times + self.phi2) + self.offset  # Heat demand [W]
        
        plt.figure()
        plt.plot(self.Q_d)
        plt.title('Heat demand consumer')
    
    def __repr__(self):
        return f"Consumer(consumer_id={self.consumer_id}, A1={self.A1}, A2={self.A2}, Period1={self.Period1}, Period2={self.Period2}, phi1={self.phi1}, phi2={self.phi2})"

    def get_heat_demand(self, N) -> float:
        """
        Calculate the heat demand of the consumer at a given time.
        Args:
            - time: current time [s]

        Returns:
            - Q_demand: heat demand [W]
        """
        return self.Q_demand[N]
    
    def set_mflow(self,N,Q_supply):
        """
        Function to determine the cold side mass flow rate at time step N. 
        Based on the difference between the heat supply and the heat demand.
        #TODO: currently a placeholder function. Need to check values 
        could also do something with the setpoint in the heat exchanger controller
        """

        # DUMMY
        if Consumer.get_heat_demand(self,N) > 2*Q_supply:
            self.mflow[N] = 0.5
        else:
            self.mflow[N] = 0.1
        

if __name__ == "__main__":
    consumer = Consumer("Consumer_1", 1000, 500, 3600, 7200, 0, np.pi/4)
    time = 1800  # Example time in seconds
    heat_demand = consumer.get_heat_demand(time)
    print(consumer)