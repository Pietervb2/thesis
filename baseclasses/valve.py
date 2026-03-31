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
            - Kvs: hydraulic conductivity of fully open valve [m3/h/bar^0.5]
            - Kv0: hydraulic conductivity of closed valve at h* [m3/h/bar^0.5]

        """

        self.valve_id = valve_id
        self.pipe_id = pipe_id # the pipe to which it is connected
        self.Kvs = valve_data[0]

        if hex is not None:
            self.Kp = valve_data[1]
            self.Ki = valve_data[2]
 
        elif node is not None:
            self.T_set_overflow = valve_data[1]
            self.P_band = valve_data[2]
            self.tau = valve_data[3]
            self.steps = valve_data[4]

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
            self.h_tau = np.zeros(num_steps) # debug, get insight in behavior
            self.Kv = np.ones(num_steps)*1e-5 # a dummy variable in which way 1/Kv is not inf if we take 0 at the beginning of the simulation. 
            self.updated_temp = 0
        else:
            self.h = self.h_overflow # set the valve displacement at a constant opening
            self.Kv = self.linear_valve(self.h)
              
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
        Kv = (self.Kvs/Kv0) ** (h-1) * self.Kvs
        return Kv
    
    def linear_valve(self, h, Kvleak = False):
        """
        Returns Kv value of the valve following a linear characteristic
        based on the valve displacement h in [0, 1]. 0 means fully closed, 1 means fully open.
        """
        self.Kvs = 0.003  # [m3/h/bar^0.5]
        
        Kv0 = self.Kvs / 50.0  # same minimum as in equal-percentage
        Kvleak = self.Kvs / 2000
        
        h_star = 0.05  # lower values of h make the form of the valve deviate from the equal percentage equation.
        
        if h < h_star and Kvleak:
            Kvr = Kv0 + h_star * (self.Kvs - Kv0)
            Kv = Kvleak + h*(Kvr - Kvleak)/h_star
        else:
            Kv = Kv0 + h * (self.Kvs - Kv0)
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
                    T_set_point = 60 # Temperature set point for the tapwater outlet [C] (brochure)
                    dT = (T_set_point - self.hex.consumer.Tc_out[N-1])
                    
                    self.I += dT * self.dt
                    delta_h = self.Kp * dT + self.Ki * self.I

                    h_star = 0.05 # lower values of h make the form of the valve deviate from the equal percentage equation.
                    
                    h = self.h[N-1] + delta_h
                    h = min(1,max(h_star,h)) # As h is scaled to hstar-1

                    # Anti-windup 
                    if (h == 0 and dT < 0) or (h == 1 and dT > 0):
                        self.I -= dT * self.dt  # unwind integral

                    Kv = self.equal_percentage_valve(h)

                    self.h[N] = h
                    self.Kv[N] = Kv
                    self.I_array[N] = self.I
                else:
                    # no change in valve position
                    self.h[N] = 0
                    self.Kv[N] = 0
            
            if self.node is not None:
                
                # stating whether you already give it a predefined position
                if self.h_overflow is None:
                    
                    # Initialization
                    if N == 0:
                        self.T_sensor[N] = 20 # initial temperature of sensor, set to T_ambt
                    else:
                        self.T_sensor[N] = self.T_sensor[N-1] + self.dt / self.tau * (self.node.T[N-1] - self.T_sensor[N-1]) # simple model for heat transfer to sensor
                    
                    sensor_temp = self.T_sensor[N]

                    if sensor_temp <= self.T_set_overflow - self.P_band/2:  # Overflow temperature set point
                        h_band = 1
                    elif sensor_temp > self.T_set_overflow + self.P_band/2:
                        h_band = 0 
                    else:

                        if self.steps == 'con':
                            h_band = (self.T_set_overflow + self.P_band/2 - sensor_temp)/self.P_band
                        else:
                            # Convert continuous P_band to discrete steps
                            h_band = np.floor((self.T_set_overflow + self.P_band/2 - sensor_temp) / self.P_band * self.steps) / self.steps
               
                    self.h_band[N] = h_band
                    
                    # tau_h = 5  # time constant for smoothing [s]
                    # h_tau = self.h[N-1] + (h_band - self.h[N-1]) * self.dt/tau_h  # smooth the changes

                    # if h_tau < 0.005:
                    #     h_tau = 0
                    h_tau = h_band
                    self.h_tau[N] = h_tau

                    Kvleak_flag = True

                    if not Kvleak_flag:
                        h_star = 0.05 # lower values of h make the form of the valve deviate from the equal percentage equation.
                        if h_tau < h_star:
                            h = 0
                        else:
                            h = min(1,h_tau)
                    else:
                        h = max(0,min(1,h_tau)) 

                    self.Kv[N] = self.linear_valve(h, Kvleak_flag)
                    self.h[N] = h
                  

# 26-3-2026 12u
# Initialization
                    # if N == 0:
                    #     node_temp = self.node.T[N]
                    #     self.T_sensor[N] = 20 # initial temperature of sensor, set to T_ambt
                    # else:
                    #     node_temp = self.node.T[N-1]
                    #     mflow = next(iter(self.node.pipes_in.values())).get_mflow(N-1) # assuming only one incoming pipe, which is the case in our network for overflow      
                       
                    #     # Heat transfer coefficient between the node and the sensor 
                    #     K = 0.5 # [J / kg K]
                    #     if mflow == 0:
                    #         mflow = 0.001
                    #     self.T_sensor[N] = self.T_sensor[N-1] + self.dt * K * mflow * (-self.T_sensor[N-1] + node_temp) # simple model for heat transfer to sensor

                    # sensor_temp = self.T_sensor[N]

                    # if sensor_temp < self.T_set_overflow - self.P_band:  # Overflow temperature set point
                    #     h_band = 1
                    # elif sensor_temp > self.T_set_overflow:
                    #     h_band = 0 
                    # else:
                        
                    #     e = self.T_sensor[N] - self.updated_temp
                    #     # Prevent very small changes in h
                    #     if abs(e) < 0.1: 
                    #         h = self.h[N-1] # keep the same valve position
                    #         self.Kv[N] = self.Kv[N-1] # keep the same valve position
                    #         return

                    #         # Convert continuous P_band to discrete steps
                    #         # num_steps_band = 20  # Define number of discrete steps
                    #         # h_band = np.floor((self.T_set_overflow - node_temp) / self.P_band * num_steps_band) / num_steps_band
                    #     h_band = (self.T_set_overflow - sensor_temp)/self.P_band

                    # self.updated_temp = self.T_sensor[N]
                    # tau = 30  # time constant for smoothing [s]
                    # h_tau = self.h[N-1] + (h_band - self.h[N-1]) * self.dt/tau  # smooth the changes
                    
                    # if h_tau < 0.01:
                    #     h_tau = 0
                    # self.h_band[N] = h_band
                    # self.h_tau[N] = h_tau

                    # Kvleak = True
                    # if not Kvleak:
                    #     h_star = 0.05 # lower values of h make the form of the valve deviate from the equal percentage equation.
                    #     if h_tau < h_star:
                    #         h = 0
                    #     else:
                    #         h = min(1,h_tau)
                    # else:
                    #     h = max(0,min(1,h_tau)) 

                    #     self.Kv[N] = self.linear_valve(h, Kvleak)
                         
                    # # Round h as very small changes don't make sense

                    # self.h[N] = h