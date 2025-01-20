from .node import Node
from .pipe import Pipe
import numpy as np


class Network:
    """
    Class used to build the network
    """
    
    def __init__(self):
        """
        Initialize an empty network.
        """
        self.nodes = {}
        self.edges = {}
        # self._next_node_id = 0
        # self._next_edge_id = 0

    def add_node(self, node_id, x, y, z, data = None) -> None:
        """
        Add a node to the network.

        # TODO: data argument can be used to later tell more about the node and what type it is. 
        """
        
        node = Node(x,y,z)
        self.nodes[node_id] = {'id': node_id, 'data': node}

    def add_pipe(self, pipe_id, from_node, to_node, data = None) -> None:
        """
        Add an pipe between two nodes.
        # TODO: data argument can be used to later tell more about the pipe, 
        # maybe easy to create a json file from where all the standard values for a pipe can be read. Like inner and outer diameter. 
        """
        if from_node not in self.nodes:
            raise ValueError(f"{from_node} must exist in the network")
        elif to_node not in self.nodes:
            raise ValueError(f"{to_node} must exist in the network")
        
        L = self.pipe_length(self.nodes[from_node], self.nodes[to_node])
        pipe = Pipe(L,0.1, 0.08, 100) # DUMMY VARIABLES TODO: find a way to make this more convient

        # pipe_id = self._next_pipe_id
        self.pipes[pipe_id] = {
            'id': pipe_id,
            'from': from_node,
            'to': to_node,
            'pipe': pipe
        }

        # self._next_pipe_id += 1
    
    def pipe_length(Node1, Node2):
        """
        Function that calculates the length of the pipe between two nodes. 
        """
        return np.sqrt((Node1.x - Node2.x)**2 + (Node1.y - Node2.y)**2 + (Node1.z - Node2.z)**2)
    
    # Functions written bij copilot. Maybe useful later on. 

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

    # def get_neighbors(self, node_id):
    #     """Get all neighboring nodes of a given node."""
    #     if node_id not in self.nodes:
    #         return []
        
    #     neighbors = []
    #     for edge in self.edges.values():
    #         if edge['from'] == node_id:
    #             neighbors.append(edge['to'])
    #         elif edge['to'] == node_id:
    #             neighbors.append(edge['from'])
    #     return neighbors