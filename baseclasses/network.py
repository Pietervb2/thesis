from node import Node
from pipe import Pipe

import numpy as np
import matplotlib.pyplot as plt

from typing import Union


class Network:
    """
    Class used to build the network
    """
    
    def __init__(self, net_id):
        """
        Initialize an empty network.
        """
        self.nodes = {}
        self.pipes = {}
        self.net_id = net_id

    def add_node(self, node_id : str, x : float, y : float , z : float, data = None) -> None:
        """
        Add a node to the network.

        # TODO: data argument can be used to later tell more about the node and what type it is. 

        Parameters:
        node_id : name of node
        x, y, z : coordinates of the nodes location
        data : additional data about the node
        """

        if node_id in self.nodes.keys():
            raise ValueError(f"Node with id {node_id} already esxists in the network")
        
        node = Node(x, y, z, node_id)
        self.nodes[node_id] = node

    def add_pipe(self, pipe_id : str, from_node : str, to_node : str, pipe_data : list) -> None:
        """
        Add a pipe between two nodes.
        # TODO: data argument can be used to later tell more about the pipe, 
        # maybe easy to create a json file from where all the standard values for a pipe can be read. Like inner and outer diameter. 
        """
        if pipe_id in self.pipes.keys():
            raise ValueError(f"Pipe with id {pipe_id} already exists in the network")

        if from_node not in self.nodes:
            raise ValueError(f"{from_node} must exist in the network")
        elif to_node not in self.nodes:
            raise ValueError(f"{to_node} must exist in the network")

        L = self.pipe_length(self.nodes[from_node], self.nodes[to_node])
        pipe = Pipe(pipe_id, L, pipe_data[0], pipe_data[1], pipe_data[2], pipe_data[3], pipe_data[4], pipe_data[5], pipe_data[6], pipe_data[7])

        # pipe_id = self._next_pipe_id
        self.pipes[pipe_id] = {
            'from': from_node,
            'to': to_node,
            'pipe_instance': pipe
        }

        # adding the pipes to the nodes
        out_node = self.nodes[from_node]
        out_node.connect_pipe_to_node(pipe_id, pipe, 'outgoing') 

        in_node = self.nodes[to_node]
        in_node.connect_pipe_to_node(pipe_id, pipe, 'incoming') 

        # self._next_pipe_id += 1
    
    def pipe_length(self, Node1, Node2):
        """
        Function that calculates the length of the pipe between two nodes. 
        """
        return np.sqrt((Node1.x - Node2.x)**2 + (Node1.y - Node2.y)**2 + (Node1.z - Node2.z)**2)
    
    def get_neighbor_nodes(self, node_id : str) -> tuple[Union[dict]]:
        """
        Get all attached nodes of a node.
        
        Parameters:
        self: Network instance
        node_id : name of the node

        Returns:
        downstream : dictionary of downstream nodes
        upstream: dictionary of upstream nodes
        """
        
        if node_id not in self.nodes:
            raise ValueError(f"Node with id: {node_id} does not exist in the network")
        
        downstream = {}
        upstream = {}

        for _, pipe in self.pipes.items():
            if pipe['from'] == node_id:
                downstream[node_id] = self.nodes[node_id]
            elif pipe['to'] == node_id:
                upstream[node_id] = self.nodes[node_id]

        return downstream, upstream
    
    def get_attached_pipes(self, node_id) -> tuple[Union[dict]]:
        """Get all attached pipes of a node.
        
        Parameters:
        self: Network instance
        node_id : name of the node

        Returns:
        incoming : dictionary of incoming pipes
        outgoing: dictionary of outgoing pipes
        """

        if node_id not in self.nodes:
            raise ValueError(f"Node with id: {node_id} does not exist in the network")
        
        incoming = self.nodes[node_id].pipes_in
        outgoing = self.nodes[node_id].pipes_out

        # TODO: vraag me af of dit goed gaat

        return incoming, outgoing
    
    def initialize_network(self, 
                           dt : float, 
                           num_steps : int, 
                           v_init_array : np.ndarray[Union[float]], 
                           T_in : np.ndarray[Union[float]],
                           T_init_water : float,
                           T_init_pipe : float,
                           file: str):
        """
        Initialize the temperature in the network

        # NOTE: for now just visit all nodes and pipes. They are all given the initial temperature of the first node. 
        """

        for i, node in enumerate(self.nodes.values()):
            
            if i == 0:
                self.nodes['Node 1'].T = T_in
            else:
                node.initialize_node(num_steps, T_init_water)

        for pipe in self.pipes.values():
            pipe['pipe_instance'].bnode_init(dt, num_steps, v_init_array, T_in, T_init_water, T_init_pipe, file)
            

    def set_T_and_flow_network(self, T_ambt : float, N : int):
            
            self.pipes_finished = []

            # Done by hand as no inflow pipe connected to node
            pipe1 = self.pipes['Pipe 1']['pipe_instance']
            pipe1.bnode_method(T_ambt, N)
            
            self.pipes_finished.append("Pipe 1")
            next_node_id = self.pipes["Pipe 1"]['to']
            next_node = self.nodes[next_node_id]
            next_node.set_T(N)
            next_node.set_m_flow(N)
            

            self.set_T_and_flow_network_rec(next_node, next_node_id, T_ambt, N)
            
    def set_T_and_flow_network_rec(self, node : Node, node_id : str, T_ambt : float, N : int):
        # print(f'Current node = {node_id}') #debug

        for pipe_id, pipe in node.pipes_out.items():

            pipe.bnode_method(T_ambt, N)
            self.pipes_finished.append(pipe_id)
            
            next_node_id = self.pipes[pipe_id]['to']
            next_node = self.nodes[next_node_id]
            # print(f' pipes finished = {self.pipes_finished}') #debug
            # print(f' next node id = {next_node_id}')    #debug

            if all(pipes in self.pipes_finished for pipes in list(next_node.pipes_in.keys())):
                # print(f'Activate {next_node}') #debug
                next_node.set_m_flow(N) # here the inlet values for the mass flow is set
                next_node.set_T(N)      # here the inlet temperature for the pipe is set coming from the node. 

                self.set_T_and_flow_network_rec(next_node, next_node_id, T_ambt, N)

    def get_total_network_length(self):
        """
        Returns the sum of all pipe lengths in the network.
        Assumes each pipe instance has an attribute 'L' for length.
        """
        return sum(pipe_data['pipe_instance'].L for pipe_data in self.pipes.values())  
if __name__ == "__main__":
    pass 
    
















