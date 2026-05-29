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
            self.max_rate = valve_data[3]
 
        elif node is not None:
            self.T_set_overflow = valve_data[1]
            self.P_band = valve_data[2]
            self.tau = valve_data[3]
            self.max_rate = valve_data[4]
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
            self.h_band = np.zeros(num_steps) # debug, get insight in behavior
            self.tau_array = np.zeros(num_steps) # debug, get insight in behavior
            self.Kv = np.ones(num_steps)*1e-5 # a dummy variable in which way 1/Kv is not inf if we take 0 at the beginning of the simulation. 
        else:
            self.h = self.h_overflow # set the valve displacement at a constant opening
            self.Kv = self.linear_valve(self.h)
              
        self.dt = dt # For PI controller

        if self.hex is not None: 
            self.I_array = np.zeros(num_steps)
            self.I = 0  # Integral term for PI controller

    def equal_percentage_valve(self, h, Kvleak_bool):
        """
        Returns Kv value of the valve following the formula for an equal percentage valve, based on the valve displacement.
        For valve displacement lower than h_star Kv does not follow the standard form of the function. It becomes unpredictable. For this region we simply assume a linear behavior.
        """        
        Kv0 = self.Kvs / 25  # same minimum as in equal-percentage
        Kvleak = self.Kvs / 2000

        h_star = 0.02

        if h < h_star and Kvleak_bool:
            Kvr = (self.Kvs/Kv0) ** (h_star-1) * self.Kvs
            Kv = Kvleak + h*(Kvr - Kvleak)/h_star
        else:
            Kv = (self.Kvs/Kv0) ** (h-1) * self.Kvs

        return Kv
    
    def linear_valve(self, h, Kvleak_bool):
        """
        Returns Kv value of the valve following a linear characteristic
        based on the valve displacement h in [0, 1]. 0 means fully closed, 1 means fully open.
        """
        # self.Kvs = 0.003  # [kg/s/Pa^0.5]
        
        Kv0 = self.Kvs / 25  # same minimum as in equal-percentage
        Kvleak = self.Kvs / 2000
        
        h_star = 0.02  # lower values of h make the form of the valve deviate from the equal percentage equation.
        
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

        minimal_opening = 0

        # P-band logic
        if self.T_sensor[N] <= self.T_set_overflow -  self.P_band:  
            h_band = 1
        elif self.T_sensor[N] > self.T_set_overflow:
            h_band = minimal_opening 
        else:
            # Choice between continuous and discrete valve displacement 
            if self.steps == "con":
                h_band = (self.T_set_overflow - self.T_sensor[N])/self.P_band + minimal_opening
            else:
                h_band = np.floor((self.T_set_overflow - self.T_sensor[N]) / self.P_band * self.steps) / self.steps + minimal_opening
        
        self.h_band[N] = h_band
        h_previous = self.h[N-1] if N > 0 else 1 # Assume fully open at the start
        h = h_previous + np.clip((h_band - h_previous), -self.max_rate*self.dt, self.max_rate*self.dt)
            
        if not Kvleak_bool:
            h_star = 0.02 # lower values of h make the form of the valve deviate from the equal percentage equation.
            if h < h_star:
                h = minimal_opening
            else:
                h = min(1,h)
        else:
            h = max(minimal_opening,min(1,h)) 

        Kv = self.linear_valve(h, Kvleak_bool)

        return h, Kv

    def benchmark_valve(self, N):
        """
        Theoretical benchmark for bypass valve including a deadband.
        """
        
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
        
        Kv = self.linear_valve(h, self.Kvleak_bool)

        return h, Kv
    
    def hex_valve(self, N, Kvleak_bool):
        """
        PI controller for the heat exchanger valve. 
        The valve position is updated based on the consumer outlet temperature.
        """
        
        T_set_point = 55 # Temperature set point for the tapwater outlet [C] (brochure)
        dT = (T_set_point - self.hex.consumer.Tc_out[N-1])
        
        self.I += dT * self.dt
        delta_h = self.Kp * dT + self.Ki * self.I

        h_previous = self.h[N-1] if N > 0 else 0 # Assume fully closed at the start
        h = h_previous + np.clip(delta_h, -self.max_rate*self.dt, self.max_rate*self.dt)

        h = max(0, min(1, h))  # Ensure h is between 0 and 1
        
        # Anti-windup 
        if (h == 0 and dT < 0) or (h == 1 and dT > 0):
            self.I -= dT * self.dt  # unwind integral

        Kv = self.equal_percentage_valve(h, Kvleak_bool)

        return h, Kv
      
    def update_valve(self, N):
        """
        Update the valve position based on the consumer outlet temperature using a PI controller.
        """
           
        # Heat exchanger valves
        if self.hex is not None:
            
            Kvleak_bool = False

            # Only update valve position if there is flow on consumer side
            if self.hex.consumer.mflow[N] > 0:

                h, Kv = self.hex_valve(N, Kvleak_bool)
                self.h[N] = h
                self.Kv[N] = Kv
                self.I_array[N] = self.I
            else:
                # no demand, so closes the valve
                # h_previous = self.h[N-1] if N > 0 else 0 # Assume fully open at the start
                # h = max(0, h_previous - self.max_rate*self.dt)

                h = 0
                self.h[N] = h
                self.Kv[N] = self.equal_percentage_valve(h, Kvleak_bool)
        
        # Overflow valve
        if self.node is not None:

            # Stating whether you already give it a predefined position
            if self.h_overflow is None:
                
                # Sensor temperature model
                if N == 0:
                    self.T_sensor[N] = 20 # initial temperature of sensor, set to T_ambt
                else:
                    
                    # pipe = self.node.pipes_out[self.pipe_id]
                    # mflow = pipe.mflow[N-1]

                    # self.mflow_ref = 500 / 3600 # kg/s (vanuit gaande dat dit de maximale flow is die het gaat bereiken)
                    # self.tau_min = 30
                    # self.tau_max = 180
                    # self.tau_nat = 180

                    # self.tau_0 = self.tau_min

                    # # print(f' mflow {mflow}')
                    # if mflow > 1e-4:  # flowing
                    #     # print(f'gaat er door heen')
                    #     tau = self.tau_0 * (self.mflow_ref / mflow) ** 0.8
                    #     tau = np.round(np.clip(tau, self.tau_min, self.tau_max),2)

                    # else:  # stagnant, natural convection in water
                    #     tau = self.tau_nat  # ~300-600 s
                    
                    tau = self.tau
                    self.T_sensor[N] = (self.T_sensor[N-1] 
                                        + self.dt / tau 
                                        * (self.node.T[N-1] - self.T_sensor[N-1]))

                    self.tau_array[N] = tau

                if self.opt:
                    h, Kv = self.BO_overflow_valve(self.Kvleak_bool, N)
                else:
                    h, Kv = self.benchmark_valve(N)

                self.h[N] = h
                self.Kv[N] = Kv
            
            else:
                # In case you only want to set the overflow to a certain opening.
                self.h[N] = self.h_overflow
                self.Kv[N] = self.linear_valve(self.h_overflow, self.Kvleak_bool)
                
