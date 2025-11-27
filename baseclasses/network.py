from node import Node
from pipe import Pipe
from heatexchanger import HeatExchanger
from pump import Pump

from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



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
        self.hexs = {}
        self.pumps = {}
        self.valves = {}

        self.net_id = net_id

    def __repr__(self):
        return f"Network(net_id={self.net_id}, nodes={list(self.nodes.keys())}, pipes={list(self.pipes.keys())}, hexs={list(self.hexs.keys())})"

    def initialize_network(self, 
                           dt : float, 
                           num_steps : int, 
                           T_in : np.ndarray[Union[float]],
                           T_init_water : float,
                           T_init_pipe : float):
        """
        Initialize the temperature in the network. 
        Loading the inlet temperature and inlet flow for the first node and pipe. The rest will be dummy variables.

        # NOTE: for now just visit all nodes and pipes. They are all given the initial temperature of the first node. 
        """
        ### Hydraulics

        # Determine incidence and loop matrix
        self.build_incidence_matrix()
        self.build_loop_matrix_from_incidence()
        self.build_help_vectors_NR()

        # Big matrix to save all the pipe mflows for faster acces to flows
        self.Q_all = np.zeros([len(self.pipe_map),num_steps])

        ### Thermodynamics

        # Initialize temperature arrays within nodes. And set input temperature for first node.
        for i, node in enumerate(self.nodes.values()):
            
            if i == 0:
                self.nodes['Node 1'].T = T_in
            else:
                node.initialize_node(num_steps, T_init_water, dt)

        # Initialize temperature and mass flow arrays in pipes. For now just pass v_inflow with a low value.
        # v_flow_history determines the velocity of the water before the simulation. 

        v_inflow = 0.004 # m/s, dummy variable chosen very small on purpose so you have a enough history. 

        for i, pipe in enumerate(self.pipes.values()):
            
            if i == 0:
                pipe['pipe_instance'].bnode_init(dt, num_steps, T_init_water, T_init_pipe, v_inflow, T_in = T_in)
            else:
                pipe['pipe_instance'].bnode_init(dt, num_steps, T_init_water, T_init_pipe, v_inflow)      
      
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
        """
        if pipe_id in self.pipes.keys():
            raise ValueError(f"Pipe with id {pipe_id} already exists in the network")

        if from_node not in self.nodes:
            raise ValueError(f"{from_node} must exist in the network")
        elif to_node not in self.nodes:
            raise ValueError(f"{to_node} must exist in the network")

        delta_z, L = self.pipe_length(self.nodes[from_node], self.nodes[to_node])
        pipe = Pipe(pipe_id, L, delta_z, pipe_data)

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

    def add_hex(self, node_id : str, from_node : str, to_node : str, hex_data : list, pipe_data : list, consumer : object) -> None:
        """
        Add a heat exchanger between two nodes.
        """
        if node_id in self.hexs.keys():
            raise ValueError(f"Heat Exchanger with id {node_id} already exists in the network")

        if from_node not in self.nodes:
            raise ValueError(f"{from_node} must exist in the network")
        elif to_node not in self.nodes:
            raise ValueError(f"{to_node} must exist in the network")
        
        # Place heat exchanger in the middle of the two nodes
        x = (self.nodes[from_node].x + self.nodes[to_node].x) / 2
        y = (self.nodes[from_node].y + self.nodes[to_node].y) / 2
        z = (self.nodes[from_node].z + self.nodes[to_node].z) / 2

        hex = HeatExchanger(x,y,z,node_id, hex_data[0], hex_data[1], hex_data[2], hex_data[3], consumer)

        # Attach pipes to the heat exchanger and the nodes
        pipe_in_id = f'Pipe {node_id.split()[-1]}.1'
        pipe_out_id = f'Pipe {node_id.split()[-1]}.2'
        
        self.nodes[node_id] = hex
        self.hexs[node_id] = hex
        
        self.add_pipe(pipe_in_id, from_node, node_id, pipe_data) # this pipe will contain a valve controlled by HEX
        self.add_pipe(pipe_out_id, node_id, to_node, pipe_data)
       
  
    def add_pump(self, pump_id : str, from_node : str, to_node : str, pipe_data : list, dp_pump) -> None:
        """
        Add pump between nodes. The pump is a pipe with an extra pressure element. 
        """
        if pump_id in self.pipes.keys():
            raise ValueError(f"Pump with id {pump_id} already exists in the network")

        if from_node not in self.nodes:
            raise ValueError(f"{from_node} must exist in the network")
        elif to_node not in self.nodes:
            raise ValueError(f"{to_node} must exist in the network")

        delta_z, L = self.pipe_length(self.nodes[from_node], self.nodes[to_node])
        pump = Pump(pump_id, L, delta_z, pipe_data, dp_pump)

        self.pipes[pump_id] = {
            'from': from_node,
            'to': to_node,
            'pipe_instance': pump
        }

        self.pumps[pump_id] = pump

        # adding the pipes to the nodes
        out_node = self.nodes[from_node]
        out_node.connect_pipe_to_node(pump_id, pump, 'outgoing') 

        in_node = self.nodes[to_node]
        in_node.connect_pipe_to_node(pump_id, pump, 'incoming') 

    def build_incidence_matrix(self):

        """
        Finding incidence matrix of the graph. Needed for Newton-Raphson implementation. 
        """
        
        # As not all nodes and pipes are named in order, first make a mapping.
        node_ids = list(self.nodes.keys())
        pipe_ids = list(self.pipes.keys())

        self.node_map = {nid: i for i, nid in enumerate(node_ids)}
        self.pipe_map = {pid: j for j, pid in enumerate(pipe_ids)}

        # Make incidence matrix and walk throw the pipes
        self.incidence_matrix = np.zeros((len(self.node_map),len(self.pipe_map)))

        for pipe_id, pipe in self.pipes.items():
            from_node = pipe['from']
            to_node = pipe['to']
            pipe_id = pipe_id

            self.incidence_matrix[self.node_map[from_node], self.pipe_map[pipe_id]] = -1
            self.incidence_matrix[self.node_map[to_node], self.pipe_map[pipe_id]] = 1
    
    def build_loop_matrix_from_incidence(self):
        """
        Obtain the loop matrix from the incidence matrix using networkx.
        """

        G = nx.DiGraph()

        node_ids = list(self.nodes.keys())
        pipe_ids = list(self.pipes.keys())

        A = self.incidence_matrix

        # build the graph from the incidence matrix
        for e, edge_id in enumerate(pipe_ids):
            # incidence matrix column: exactly +1 / −1 at endpoints
            tail = node_ids[np.where(A[:, e] == -1)[0][0]]
            head = node_ids[np.where(A[:, e] == +1)[0][0]]
            G.add_edge(tail, head, id=edge_id, col=e)

        # compute all fundamental cycles
        cycles = nx.cycle_basis(G.to_undirected())

        # loop matrix
        self.loop_matrix = np.zeros((len(cycles), len(pipe_ids)))

        for i, cycle in enumerate(cycles):
            # convert cycle (list of nodes) into list of directed edges
            for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
                if G.has_edge(u, v):      # aligned edge
                    e = G[u][v]['col']
                    self.loop_matrix[i, e] = +1
                elif G.has_edge(v, u):    # opposite direction edge, but set it to +1 as all directions are fixed
                    e = G[v][u]['col']
                    self.loop_matrix[i, e] = +1

    def build_help_vectors_NR(self):

        """
        Build head difference vectors for friction and elevation per pipe. And pump vector.
        """
    
        self.pressure_friction_vector = np.zeros(len(self.pipes))
        self.pressure_elevation_vector = np.zeros(len(self.pipes))
       
        for j, pipe_id in enumerate(self.pipes.keys()):

            pipe = self.pipes[pipe_id]['pipe_instance']
            self.pressure_friction_vector[j] = pipe.pressure_friction()
            self.pressure_elevation_vector[j] = pipe.pressure_elevation()

        # Vector of pump pressure per pipe
        self.pump_array = np.zeros(len(self.pipes))
        for pump in self.pumps.values():
            j = self.pipe_map[pump.pipe_id]
            self.pump_array[j] = pump.pressure_head()

        # Vector of pressure per HEX. Which will be mapped to the inlet pipe of the HEX.
        self.hex_array = np.zeros(len(self.pipes))
        for hex_obj in self.hexs.values():
            pipe_id, pipe_obj = next(iter(hex_obj.get_incoming_pipes().items()))
            j = self.pipe_map[pipe_id]
            self.hex_array[j] = hex_obj.pressure_drop() / (pipe_obj.A**2 * 1e3)


    def set_mflow_network(self, N : int):

        """
        Impelement Newton-Raphson to solve for mass flows in the network.

        Steps:
        1. Set up initial guess for mass flows in all pipes
        2. Reduce the incidence matrix by removing the first row (reference node)
        3. Set up the residual vector F(Q) containing continuity equations and loop head-loss equations
        4. Set up the Jacobian matrix J
        5. Solve for update in mass flows dQ: J dQ = -F(Q)
        6. Update mass flows: Q = Q + dQ

        """
        # prepare pipe and loop data
        pipe_ids = list(self.pipes.keys())
        p = len(pipe_ids)

        # Initial guess for the mflows in the pipes
        Q0 = np.zeros(p)
        if N == 0:
            Q0[:] = 0.01
        else:
            Q0 = self.Q_all[:,N-1]    

        incidence_matrix_reduced = self.incidence_matrix[1:,:]  # Remove first row to account for reference node

        # Newton-Raphson
        error = 0
        tolerance = 1e-6
        mflow = Q0.copy()
        max_iter = 100     

        for it in range(max_iter):

            # Continuity and loop head-loss equations
            continuity = incidence_matrix_reduced @ mflow

            friction_vector = self.pressure_friction_vector + self.hex_array  # Add pressure drop of HEXs
            head_loss  = self.loop_matrix @ (friction_vector * mflow**2
                                            + self.pressure_elevation_vector)

            # Pump contribution (only in loop equations)
            pump_term = self.loop_matrix @ self.pump_array

            # Full residual vector
            F = np.concatenate([continuity, head_loss - pump_term])
            
            error = np.linalg.norm(F)
            if error < tolerance:
                break
            
            # Jacobian
            J = np.vstack([incidence_matrix_reduced, self.loop_matrix * (2 * self.pressure_friction_vector * mflow)])  

            # solve for delta y: J dy = -F
            try:
                dmflow = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # fallback to least squares / pseudo-inverse
                dmflow = np.linalg.lstsq(J, -F, rcond=None)[0]
                print("Warning: Jacobian is singular, using least squares solution.")

            mflow += dmflow
        
        if error >= tolerance:
            raise RuntimeError(f"Newton-Raphson did not converge within {max_iter} iterations, final error: {error}")
        
        self.Q_all[:,N] = mflow

        # Assign flows to the pipes for timestep N
        for j, pipe_id in enumerate(pipe_ids):
            pipe = self.pipes[pipe_id]['pipe_instance']
            pipe.set_mflow(mflow[j],N)

    def set_T_network(self, T_ambt : float, N : int, no_cap = False):
            
            """"
            Set temperature in the network.
            """

            # List of pipes for which the bnode method is performed.
            self.pipes_finished = []

            # Done by hand as no inflow pipe connected to node, gets Pipe 1. But now name is not hard coded.
            first_pipe = next(iter(self.pipes.values()))['pipe_instance'] 
            
            # first_pipe.set_T_in(first_node.T[N], N) # Set inlet temperature
            first_pipe.bnode_method(T_ambt, N, no_cap = no_cap)

            self.pipes_finished.append(first_pipe.pipe_id)

            # Get node at end of pipe. 
            next_node_id = self.pipes[first_pipe.pipe_id]['to']
            next_node = self.nodes[next_node_id]

            # Update temperature 
            next_node.set_T(N)
            
            self.set_T_network_rec(next_node, next_node_id, T_ambt, N, no_cap = False)
          
    def set_T_network_rec(self, node : Node, node_id : str, T_ambt : float, N : int, no_cap = False):
        """"
        To determine which node to update next I perform a recursion. 
        As for all incoming pipes the bnode method needs to be completed before the node is updated. 
        """

        for pipe_id, pipe in node.pipes_out.items():

            pipe.bnode_method(T_ambt, N, no_cap = no_cap)
            self.pipes_finished.append(pipe_id)
            
            next_node_id = self.pipes[pipe_id]['to']
            next_node = self.nodes[next_node_id]

            if all(pipes in self.pipes_finished for pipes in list(next_node.pipes_in.keys())):

                if next_node_id == 'Node 1':
                    continue    # Skip first node as its temperature is already set.

                next_node.set_T(N)      # here the inlet temperature for the pipe is set coming from the node. 
                self.set_T_network_rec(next_node, next_node_id, T_ambt, N, no_cap = no_cap)    
        
    def pipe_length(self, Node1, Node2):
        """
        Function that calculates the length of the pipe between two nodes. 
        """
        return Node2.z - Node1.z, np.sqrt((Node1.x - Node2.x)**2 + (Node1.y - Node2.y)**2 + (Node1.z - Node2.z)**2)
    
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

        return incoming, outgoing
    
    def get_total_network_length(self):
        """
        Returns the sum of all pipe lengths in the network.
        Assumes each pipe instance has an attribute 'L' for length.
        """
        return sum(pipe_data['pipe_instance'].L for pipe_data in self.pipes.values())  


if __name__ == "__main__":
    
    pass
















