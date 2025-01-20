import numpy as np
import matplotlib.pyplot as plt

class Node:

    def __init__(self, x, y, z):

        self.edges_in = []
        self.edges_out = []
        self.T = 0 # [K] temperature of the node
        
        self.x = x
        self.y = y
        self.z = z  

        """
        TODO: Working with T and flow requires using a timestep. This can be later implemented when implementing the for loop simulation. 
        """


    def add_edge(self, edge, direction):
        """
        Function that adds an edge to the node.

        Args:
            edge (Edge): the edge to be added to the node.
            direction (str): the direction of the edge, either 'in' or 'out'.
        """

        if direction == 'in':
            self.edges_in.append(edge)
        elif direction == 'out':
            self.edges_out.append(edge)
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
    def remove_edge(self,edge):
        """
        Function that removes an edge from the node.

        Args:
            edge (Edge): the edge to be removed from the node.
        """

        if edge in self.edges_in:
            self.edges_in.remove(edge)
        elif edge in self.edges_out:
            self.edges_out.remove(edge)
        else:
            raise ValueError(f"Edge {edge} not found in node")

    def set_flow(self):
        """
        Function that calculates the flow of each in- and output edge of the node.

        For now divide the mass flow based on the diameter of the pipes. 
        """         

        # total mass in flow
        total_m_inflow = sum(edge.m_flow for edge in self.edges_in)

        # assume complete filling of pipe and simply divide total mass flow #NOTE: check this assumption
        m_outflow = total_m_inflow / len(self.edges_in)

        for edge in self.edges_out:
            edge.m_flow = m_outflow      

        # TODO: workout the pressure calculation to further give this meaning. 

    def set_T(self):
        """
        Function that calculates the temperature at the node

        TODO: for now it just uses the weighted average based on the in- and outflow. 
        """
        
        sum_T_flow = 0
        sum_m_flow = 0
        for edge in self.edges_in:
            sum_T_flow += edge.T * edge.m_flow 
            sum_m_flow += edge.m_flow
        
        self.T = sum_T_flow / sum_m_flow

    def get_T(self):
        return self.T

    def get_number_edges_in(self):
        return len(self.edges_in)

    def get_number_edges_out(self):
        return len(self.edges_out)
    
