import numpy as np

class Valve:

    def __init__(self,
                 valve_id: str,
                 pipe_id: str,
                 valve_data : list,
                 h_initial : float,                 
                 hex = None,
                 node = None):
        """"
        Args:
            - Kvs: hydraulic conductivity of fully open valve [m3/h/bar^0.5]
            - Kv0: hydraulic conductivity of closed valve at h* [m3/h/bar^0.5]

        """

        self.valve_id = valve_id
        self.pipe_id = pipe_id # the pipe to which it is connected
        self.Kvs = valve_data[0]

        if hex is not None:
            self.Ki = valve_data[1]
            self.Kp = valve_data[2] 
        elif node is not None:
            self.T_set_overflow = valve_data[1]        

        self.hex = hex  # Heat exchanger controlling the valve
        self.node = node  # Node located at incoming side of pipe incase of overflow valve.

        self.h_initial = h_initial # Initial position of valve
    
    def initialize_valve(self, num_steps, dt):
        """
        Initialize the valve parameters.
        """ 

        self.h = np.ones(num_steps)*self.h_initial
        self.Kv = np.ones(num_steps)*self.equal_percentage_valve(self.h_initial)

        self.dt = dt # For PI controller

        if self.hex is not None: 
            self.I_array = np.zeros(num_steps)
            self.I = 0  # Integral term for PI controller


    def equal_percentage_valve(self, h):
        """
        Returns Kv value of the valve following the formula for an equal percentage valve, based on the valve displacement.
        For valve displacement lower than h_star Kv does not follow the standard form of the function. It becomes unpredictable. For this region we simply assume a linear behavior.
        """        
        Kv0 = self.Kvs/50
        h_star = 0.05

        # if h < h_star:
        #     Kv_star = (self.Kvs/Kv0) ** (h_star-1) * self.Kvs
        #     Kv = Kvleak + h*(Kv_star - Kvleak)/h_star
        # else:
        Kv = (self.Kvs/Kv0) ** (h-1) * self.Kvs
        return Kv
    
    def update_valve(self, N):
        """
        Update the valve position based on the consumer outlet temperature using a PI controller.
        """

        if N < len(self.h)-1: # to prevent index out of range
            
            if self.hex is not None:

                # Only update valve position if there is flow
                if self.hex.consumer.mflow[N] > 0:
                    
                    # implement PI controller to determine the valve lift

                    T_set_point = 60 # Temperature set point for the tapwater outlet [C]
                    dT = (T_set_point - self.hex.consumer.Tc_out[N])
                    
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
                    self.Kv[N+1] = 0
            
            if self.node is not None:

                # if self.node.T[N] < self.T_set_overflow:  # Overflow temperature set point
                #     self.h[N+1] = 1  # Fully open
                #     self.Kv[N+1] = self.Kvs

                # else:
                #     # no change in valve position
                #     self.h[N+1] = 0
                #     self.Kv[N+1] = 0
                
                self.h[N+1] = 1
                self.Kv[N+1] = self.Kvs


                



