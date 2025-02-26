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
        pipe = Pipe(pipe_id, L, pipe_data[0], pipe_data[1], pipe_data[2])

        # pipe_id = self._next_pipe_id
        self.pipes[pipe_id] = {
            'from': from_node,
            'to': to_node,
            'pipe_class': pipe
        }

        # adding the pipes to the nodes
        out_node = self.nodes[from_node]
        out_node.add_pipe(pipe_id, pipe, 'outgoing') 

        in_node = self.nodes[to_node]
        in_node.add_pipe(pipe_id, pipe, 'incoming') 

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
                           T_init_array : np.ndarray[Union[float]],
                           T_ambt : float):
        """
        Initialize the temperature in the network

        # NOTE: for now just visit all nodes and pipes. They are all given the initial temperature of the first node. 
        """

        for node in self.nodes.values():
            node.initialize_node(num_steps, T_init_array[0])

        for pipe in self.pipes.values():
            T_init = T_ambt # NOTE: check this 
            pipe['pipe_class'].bnode_init(dt, num_steps, v_init_array, T_init_array, T_init)

    def set_T_and_flow_network(self, T_ambt : float, v_inflow: float, T_in: float, N : int):
            
            self.pipes_finished = []
            # Done by hand as no inflow pipe connected to node
            self.nodes['Node 1'].T[N] = T_in

            pipe1 = self.pipes['Pipe 1']['pipe_class']

            pipe1.set_T_in(T_in, N)
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

    def set_T_network(self, T_ambt : float, v_inflow: float, T_in: float, N : int):
        """
        Function that determines the the mass flow for the whole network. 

        TODO: T and m flow functions are split as eventually when the implementation will become more ellaborated the determination of the mass flow will be done in a different approach to the system. 
        """
        
        self.pipes_finished = []

        # Initialize the first pipe as it does not inheret the values from the previous pipes
        pipe1 = self.pipes["Pipe 1"]['pipe_class']
        pipe1.set_T_in(T_in, v_inflow, N)

        self.pipes_finished.append("Pipe 1")

        next_node_id = self.pipes["Pipe 1"]['to']
        next_node = self.nodes[next_node_id]

        self.set_T_network_rec(next_node, next_node_id, T_ambt, N)
      
    def set_T_network_rec(self, node : Node, node_id : str, T_ambt : float, N : int):
        """
        Recurring function for walking through the network.
        """
        print(f'Current node = {node_id}') #debug

        for pipe_id, pipe in node.pipes_out.items():

            pipe.bnode_method(T_ambt, N)
            self.pipes_finished.append(pipe_id)
            
            next_node_id = self.pipes[pipe_id]['to']
            next_node = self.nodes[next_node_id]
            print(f' pipes finished = {self.pipes_finished}') #debug
            print(f' next node id = {next_node_id}')    #debug



            if all(pipes in self.pipes_finished for pipes in list(next_node.pipes_in.keys())):
                print(f'Activate {next_node}') #debug
                next_node.set_T(N) # here the initial values for the mass inflow and the temperature are set
                self.set_T_network_rec(next_node, next_node_id, T_ambt, N)

   
    def set_m_flow_network(self, ):
        """
        Function that determines the the mass flow for the whole network. 

        TODO: T and m flow functions are split as eventually when the implementation will become more ellaborated the determination of the mass flow will be done in a different approach to the system. 
        """

        pass
           
if __name__ == "__main__":
    pass 
    # Start interactive session
    # import code
    # code.interact(local=locals())

    pipe_radius_outer = 0.1 # [m] DUMMY
    pipe_radius_inner = 0.08 # [m] DUMMY
    K = 1 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
    pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

    net = Network()
    net.add_node('Node 1',0,0,0)
    net.add_node('Node 2',0,5,0)
    net.add_node('Node 3',-100,5,0)
    net.add_node('Node 4',-100,10,0)
    net.add_node('Node 5',0,10,0)
    net.add_node('Node 6',0,100,0)
    net.add_node('Node 7',50,100,0)


    net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
    net.add_pipe('Pipe 2','Node 2','Node 3', pipe_data)	
    net.add_pipe('Pipe 3','Node 3','Node 4', pipe_data)	
    net.add_pipe('Pipe 4','Node 4','Node 5', pipe_data)	
    net.add_pipe('Pipe 5','Node 2','Node 5', pipe_data)	
    net.add_pipe('Pipe 6','Node 5','Node 6', pipe_data)	
    net.add_pipe('Pipe 7','Node 6','Node 7', pipe_data)	

    net.plot_network()




















   # NOTE: Functions written bij copilot. Maybe useful later on. 

    # def remove_node(self, node_id):
    #     """Remove a node and all its connected edges."""
    #     if node_id in self.nodes:
    #         # Remove all edges connected to this node
    #         edges_to_remove = [
    #             edge_id for edge_id, edge in self.edges.items()
    #             if edge['from'] == node_id or edge['to'] == node_id
    #         ]
    #         for edge_id in edges_to_remove:
    #             del self.edges[edge_id]
            
    #         # Remove the node
    #         del self.nodes[node_id]

    # def remove_edge(self, edge_id):
    #     """Remove an edge from the network."""
    #     if edge_id in self.edges:
    #         del self.edges[edge_id]