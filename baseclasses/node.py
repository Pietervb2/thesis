import numpy as np
import matplotlib.pyplot as plt

class Node:

    def __init__(self, x, y, z, node_id):

        self.pipes_in = {}
        self.pipes_out = {}
        
        self.x = x
        self.y = y
        self.z = z  
        self.node_id = node_id 
       

    def initialize_node(self, num_steps, T_init, dt) -> None:
        """
        Initialize the temperature in the node

        Args:
        num_steps: number of steps the simulation takes
        T_init: initial temperature [K]
        dt: not used here, but in the initialize consumer method
        """

        self.T = np.zeros(num_steps)
        self.T[0] = T_init


    def connect_pipe_to_node(self, pipe_id, pipe, direction):
        """
        Function that adds an pipe to the node.

        Args:
            pipe_id : id of the pipe
            pipe (Pipe): the pipe to be added to the node.
            direction (str): the direction of the pipe, either 'in' or 'out'.
        """

        if direction == 'incoming':
            self.pipes_in[pipe_id] = pipe
        elif direction == 'outgoing':
            self.pipes_out[pipe_id] = pipe
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
    def set_T(self, N):
        """
        Sets the node temperature and the temperature of the outgoing pipes.
        """         

        sum_T_flow = 0
        sum_m_inflow = 0
        for _, pipe in self.pipes_in.items():
            
            m_inflow = pipe.get_mflow(N)
            sum_T_flow += pipe.T[N] * m_inflow
            sum_m_inflow += m_inflow
        try:    
            self.T[N] = sum_T_flow / sum_m_inflow # set the node temperature

            # Set temperature per pipe
            for _, pipe in self.pipes_out.items():
                pipe.set_T_in(self.T[N], N)

        except(ZeroDivisionError):
            print(f"No ingoing mass flow in node  = {self.node_id}")
            

    def get_T(self, N):
        return self.T[N]

    def get_number_pipes_in(self):
        return len(self.pipes_in)

    def get_number_pipes_out(self):
        return len(self.pipes_out)
    
    def get_incoming_pipes(self):
        return self.pipes_in


if __name__ == "__main__":

    pass