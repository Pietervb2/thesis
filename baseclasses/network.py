from node import Node
from pipe import Pipe
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
    
    def get_neighbor_nodes(self, node_id):
        """Get all neighbouring nodes of a node."""
        
        if node_id not in self.nodes:
            return []
        
        incoming = []
        outgoing = []
        for pipe in self.pipes.values():
            if pipe['from'] == node_id:
                outgoing.append(pipe['to'])
            elif pipe['to'] == node_id:
                incoming.append(pipe['from'])

        return incoming, outgoing
    
    def get_attached_pipes(self, node_id):
        """Get all attached pipes of a node."""

        if node_id not in self.nodes:
            return []
        
        incoming = []
        outgoing = []
        for pipe in self.pipes.values():
            if pipe['from'] == node_id:
                outgoing.append(pipe['id'])
            elif pipe['to'] == node_id:
                incoming.append(pipe['id'])

        return incoming, outgoing

    def set_T_network():
        pass

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
    net.add_pipe('Pipe 1','Node 1','Node 2')
    net.add_pipe('Pipe 2','Node 1','Node 2')	


    net.add_node('Node 3', 2,3,4)
    

    print(net.get_attached_pipes('Node 1'))
    print(net.get_neighbor_nodes('Node 1'))






















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

