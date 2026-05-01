import numpy as np

class Valve:

    def __init__(self,
                 valve_id: str,
                 pipe_id: str,
                 valve_data : list,
                 hex = None,
                 node = None,
                 h_overflow = None):
        """"
        Args:
            - Kvs: hydraulic conductivity of fully open valve [kg/s/Pa^0.5]
            - Kv0: hydraulic conductivity of closed valve at h* [kg/s/Pa^0.5]

        """

        self.valve_id = valve_id
        self.pipe_id = pipe_id # the pipe to which it is connected
        self.Kvs = valve_data[0]

        if hex is not None:
            self.Kp = valve_data[1]
            self.Ki = valve_data[2]
 
        elif node is not None:
            self.T_set_overflow = valve_data[1]
            self.Kp = valve_data[2] # P-term for overflow valve control (theta_5)
            self.Ki = valve_data[3] # I-term for overflow valve control (theta_6)
            self.tau = valve_data[4]
            self.steps = valve_data[5]
            self.Kvleak_bool = valve_data[6]
            self.opt = valve_data[7]


        self.h_overflow = h_overflow  
  
        self.hex = hex  # Heat exchanger controlling the valve
        self.node = node  # Node located at incoming side of pipe incase of overflow valve.
   
    def initialize_valve(self, num_steps, dt):
        """
        Initialize the valve parameters.
        """ 

        if self.h_overflow is None:
            self.h = np.zeros(num_steps)
            self.T_sensor = np.zeros(num_steps) # temperature of sensor
            self.h_PI_lim = np.zeros(num_steps) # debug, get insight in behavior
            self.h_tau = np.zeros(num_steps) # debug, get insight in behavior
            self.Kv = np.ones(num_steps)*1e-5 # a dummy variable in which way 1/Kv is not inf if we take 0 at the beginning of the simulation. 
            self.updated_temp = 0
        else:
            self.h = self.h_overflow # set the valve displacement at a constant opening
            self.Kv = self.linear_valve(self.h)
              
        self.dt = dt # For PI controller

  
        self.I_array = np.zeros(num_steps)
        self.I = 0  # Integral term for PI controller


    def equal_percentage_valve(self, h):
        """
        Returns Kv value of the valve following the formula for an equal percentage valve, based on the valve displacement.
        For valve displacement lower than h_star Kv does not follow the standard form of the function. It becomes unpredictable. For this region we simply assume a linear behavior.
        """        
        Kv0 = self.Kvs/50
        Kv = (self.Kvs/Kv0) ** (h-1) * self.Kvs
        return Kv
    
    def linear_valve(self, h, Kvleak_bool = False):
        """
        Returns Kv value of the valve following a linear characteristic
        based on the valve displacement h in [0, 1]. 0 means fully closed, 1 means fully open.
        """
        # self.Kvs = 0.003  # [m3/h/bar^0.5]
        
        Kv0 = self.Kvs / 50.0  # same minimum as in equal-percentage
        Kvleak = self.Kvs / 2000
        
        h_star = 0.05  # lower values of h make the form of the valve deviate from the equal percentage equation.
        
        if h < h_star and Kvleak_bool:
            Kvr = Kv0 + h_star * (self.Kvs - Kv0)
            Kv = Kvleak + h*(Kvr - Kvleak)/h_star
        else:
            Kv = Kv0 + h * (self.Kvs - Kv0)
        return Kv
    
    def BO_overflow_valve(self, Kvleak_bool, N):
        """"
        Returns the valve position and Kv value of the overflow valve based on the lumped BO model. 
        """

        # implement PI controller to determine the valve lift
        T_set_point = 55 # Temperature set point for the tapwater outlet [C] (brochure)
        dT = (T_set_point - self.T_sensor[N])
        
        self.I += dT * self.dt
        delta_h = self.Kp * dT + self.Ki * self.I

        # Max valve displacement
        max_rate = 1/300  # [%/s]
        max_step = max_rate * self.dt
        
        h_previous = self.h[N-1] if N > 0 else 0
        h_PI_lim = h_previous + np.clip(delta_h, -max_step, max_step) # new h based on PI controller, but limited to max_step to prevent too fast changes in valve position
      
        h = max(0, min(1, h_PI_lim)) # Ensure h is between 0 and 1
        
        # Anti-windup 
        if (h == 0 and dT < 0) or (h == 1 and dT > 0):
            self.I -= dT * self.dt  # unwind integral

        Kv = self.equal_percentage_valve(h)

        self.h[N] = h
        self.Kv[N] = Kv
        self.I_array[N] = self.I                
  
        self.h_PI_lim[N] = h_PI_lim
        
        if not Kvleak_bool:
            h_star = 0.05 # lower values of h make the form of the valve deviate from the equal percentage equation.
            if h_PI_lim < h_star:
                h = 0

        Kv = self.linear_valve(h, Kvleak_bool)

        return h, Kv

    def benchmark_valve(self, N):
        """
        Theoretical benchmark for bypass valve including a deadband.
        """
        self.P_band = 3 # [°C] 
        
        if N == 0:
            h = 1
        else:
            upper = self.T_set_overflow + self.P_band/2
            lower = self.T_set_overflow - self.P_band/2
            
            if self.h[N-1] == 1:
                if self.T_sensor[N] <= upper:
                    h = 1
                else:
                    h = 0
            elif self.h[N-1] == 0:
                if self.T_sensor[N] >= lower:
                    h = 0
                else:
                    h = 1
        
        Kv = self.linear_valve(h)
        return h, Kv
       
    def update_valve(self, N):
        """
        Update the valve position based on the consumer outlet temperature using a PI controller.
        """

        # if N < len(self.h)-1: # to prevent index out of range
            
        # Heat exchanger valves
        if self.hex is not None:

            # Only update valve position if there is flow on consumer side
            if self.hex.consumer.mflow[N] > 0:

                # implement PI controller to determine the valve lift
                T_set_point = 60 # Temperature set point for the tapwater outlet [C] (brochure)
                dT = (T_set_point - self.hex.consumer.Tc_out[N-1])
                
                self.I += dT * self.dt
                delta_h = self.Kp * dT + self.Ki * self.I

                h_star = 0.05 # lower values of h make the form of the valve deviate from the equal percentage equation.
                
                h = self.h[N-1] + delta_h

                if h < h_star:
                    h = 0
                else:
                    h = min(1,h) # Ensure h is between 0 and 1
                
                # Anti-windup 
                if (h == 0 and dT < 0) or (h == 1 and dT > 0):
                    self.I -= dT * self.dt  # unwind integral

                Kv = self.equal_percentage_valve(h)

                self.h[N] = h
                self.Kv[N] = Kv
                self.I_array[N] = self.I
            else:
                # no demand, so closed.
                self.h[N] = 0
                self.Kv[N] = 0
        
        # Overflow valve
        if self.node is not None:

            # stating whether you already give it a predefined position
            if self.h_overflow is None:
                
                # Sensor temperature model
                if N == 0:
                    self.T_sensor[N] = 20 # initial temperature of sensor, set to T_ambt
                else:
                    self.T_sensor[N] = self.T_sensor[N-1] + self.dt / self.tau * (self.node.T[N-1] - self.T_sensor[N-1]) # simple model for heat transfer to sensor

                if self.opt:
                    h, Kv = self.BO_overflow_valve(self.Kvleak_bool, N)
                else:
                    h, Kv = self.benchmark_valve(N)

                self.h[N] = h
                self.Kv[N] = Kv
            
            else:
                self.h[N] = self.h_overflow
                self.Kv[N] = self.linear_valve(self.h_overflow, self.Kvleak_bool)
                
