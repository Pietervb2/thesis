from node import Node
from pipe import Pipe

import numpy as np

from typing import Union


class Network:
    """
    Class used to build the network
    """
    
    def __init__(self):
        """
        Initialize an empty network.
        """
        self.nodes = {}
        self.pipes = {}
        # self._next_node_id = 0
        # self._next_edge_id = 0

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
        
        node = Node(x,y,z)
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
        pipe = Pipe(L, pipe_data[0], pipe_data[1], pipe_data[2])

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
    
    def initialize_network(self, dt : float, num_steps : int, v_init : float, T_init : float):
        """
        Initialize the temperature in the network

        # NOTE: for now just visit all nodes and pipes. They are all given the initial temperature of the first node. 
        """

        for node in self.nodes.values():
            node.T = T_init

        for pipe in self.pipes.values():
            pipe['pipe_class'].bnode_init(dt, num_steps, v_init, T_init)
            pipe['pipe_class'].calc_mass_flow(v_init)

      # TODO: hoe doe je dit nu met de nieuwe bnode_init functie? Want nu vraag je al een hele array van v_flow en T_inlet. terwijl je die nog niet hebt. 
      # Ben opzich wel tevreden met hoe het nu er uit ziet alleen moet het waarschijnlijk wat aanpassen.

    def set_T_network(self, T_ambt : float, N : int, v_in_flow: float):
        
        """  
        pak de Temperatuur van node zo heb je overal de inlet temperatuur
        dan update je de pipes
        dan update je temperatuur in de pipes
        TODO: update text and add comments
        """
        """
        pak de inlet snelheid uit de node en de temperatuur van de node, bij node 'from' 
        voer bnode methode uit
        sla de nieuwe temperatuur op in de node en gebruik de mass_flow bij node 'to' 
        """

        # Moet het hele stuk over de stroomsnelheid nog doorgeven

        for pipe in self.pipes.keys():

            node_out = self.nodes[pipe['from']]
            T_inlet = node_out.T
            pipe['pipe_class'].set_inlet_temperature(T_inlet, N)

            # Get output temperature of the pipe
            pipe["pipe_instance"].bnode_method(N, T_ambt)

        for node in self.nodes.keys():
            node_instance = self.nodes[node]
            node_instance.set_T(N)
        
        



    def set_m_flow_network():
        pass
    
    def overview_network(self):
        """
        Get overview of network. TODO: eventually it should return a plot of the network. 
        """
        pass 


if __name__ == "__main__":

    net = Network()
    net.add_node('Node 1',1,1,1)
    net.add_node('Node 2',2,2,2)
    
    # Pipe parameters
    pipe_radius_outer = 0.1 # [m] DUMMY
    pipe_radius_inner = 0.08 # [m] DUMMY
    K = 0.4 # heat transmission coefficient DUMMY zie ik staan in book van Max pagina 77
    pipe_data = [pipe_radius_outer, pipe_radius_inner, K]

    net.add_pipe('Pipe 1','Node 1','Node 2', pipe_data)
    net.add_pipe('Pipe 2','Node 1','Node 2', pipe_data)	
    
    net.get_attached_pipes('Node 1')

    # print(net.get_attached_pipes('Node 1'))
    # print(type(net.get_neighbor_nodes('Node 1')))
    
    print(f'dictionary of network of pipes before init{net.pipes}')
    net.initialize_network(0.1, 100, 2, 80)
    print(f'dictionary of network of pipes after init{net.pipes}')

    print(net.nodes['Node 1'])
    print(net.nodes['Node 2'])


    # Start interactive session
    import code
    code.interact(local=locals())


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

