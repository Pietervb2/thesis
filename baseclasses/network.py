from .node import Node
from .pipe import Pipe
from .heatexchanger import HeatExchanger
from .pump import Pump
from .valve import Valve

from scipy.optimize import root, minimize
from typing import Union

import numpy as np
import networkx as nx
import os
import pickle


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

        self.theta = None #debug

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
        ### Thermodynamics

        # Initialize temperature arrays within nodes. And set input temperature for first node.
        for i, node in enumerate(self.nodes.values()):
            
            if i == 0:
                node.initialize_node(num_steps, T_in, dt)
            else:
                node.initialize_node(num_steps, T_init_water, dt)

        # Initialize temperature and mass flow arrays in pipes. For now just pass v_inflow with a low value.
        # v_flow_history determines the velocity of the water before the simulation. 

        v_inflow = 0.004 # m/s, dummy variable chosen very small on purpose so you have a enough history. 

        for i, pipe in enumerate(self.pipes.values()):
           
            pipe['pipe_instance'].bnode_init(dt, num_steps, T_init_water, T_init_pipe, v_inflow)      

        for valve in self.valves.values():
            valve.initialize_valve(num_steps, dt)
        
        ### Hydraulics

        # Determine incidence and loop matrix
        self.build_incidence_matrix_and_graph()
        self.build_help_vectors_NR()

        # Big matrices for faster access during simulation
        self.mflow_all = np.zeros([len(self.pipe_map),num_steps])
      
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

    def add_pipe(self, pipe_id : str, from_node : str, to_node : str, pipe_data : list, hex_pipe = False) -> None:
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
           
        pipe = Pipe(pipe_id, L, delta_z, pipe_data, hex_pipe)

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

        hex = HeatExchanger(x, y, z, node_id, hex_data, consumer)

        # Attach pipes to the heat exchanger and the nodes
        pipe_in_id = f'Pipe {node_id.split()[-1]}.3'
        pipe_out_id = f'Pipe {node_id.split()[-1]}.4'
        
        self.nodes[node_id] = hex
        self.hexs[node_id] = hex
        
        self.add_pipe(pipe_in_id, from_node, node_id, pipe_data, hex_pipe = True) # this pipe will contain a valve controlled by HEX
        self.add_pipe(pipe_out_id, node_id, to_node, pipe_data, hex_pipe = True)  

        # Create valve and couple it to the hex
        valve_id = node_id.replace('Hex','Valve')
        valve_data = hex_data[-3:]
        valve = Valve(valve_id, pipe_in_id, valve_data, hex=hex)
        self.valves[valve_id] = valve
  
    def add_pump(self, pump_id : str, from_node : str, to_node : str, pipe_data : list, pump_data) -> None:
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
        pump = Pump(pump_id, L, delta_z, pipe_data, pump_data)

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

    def add_overflow(self, overflow_id : str, from_node: str, to_node: str, pipe_data: list, valve_data : list, node, h_overflow = None) -> None:
        """
        Add overflow valve at the top of the network.
        """

        if overflow_id in self.pipes.keys():
            raise ValueError(f"Overflow valve with id {overflow_id} already exists in the network")

        valve_id = 'Overflow valve'
        overflow_valve = Valve(valve_id, overflow_id, valve_data, node = node, h_overflow = h_overflow)
        self.valves[valve_id] = overflow_valve

        self.add_pipe(overflow_id, from_node, to_node, pipe_data)

    def build_incidence_matrix_and_graph(self):

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
        
        self.G = nx.DiGraph()

        node_ids = list(self.nodes.keys())
        pipe_ids = list(self.pipes.keys())

        A = self.incidence_matrix

        # build the graph from the incidence matrix
        for e, edge_id in enumerate(pipe_ids):
            # incidence matrix column: exactly +1 / −1 at endpoints
            tail = node_ids[np.where(A[:, e] == -1)[0][0]]
            head = node_ids[np.where(A[:, e] == +1)[0][0]]
            self.G.add_edge(tail, head, id=edge_id, col=e)

    def build_loop_matrix(self, graph):
        """
        Obtain the loop matrix from the graph of the network
        """
        # compute all fundamental cycles for a directed graph
        cycles = sorted(list(nx.simple_cycles(graph)), key=len) # sorted adds determinism to the order of cycles.

        # loop matrix
        self.loop_matrix_active = np.zeros((len(cycles), graph.number_of_edges()))

        # Order of columns is important for the loop matrix as the other vectors are also scaled by the mask.
        # So I force the original order of how the pipes are added on the loop matrix

        list_of_edges = []
        for u, v, data in graph.edges(data=True):
            pipe_id = data["id"]
            j = self.pipe_map[pipe_id]
            list_of_edges.append((j,pipe_id))
        
        ordered_edges = sorted(list_of_edges)

        pipe_to_index = {}
        for idx in range(len(ordered_edges)):
            edge = ordered_edges[idx][1]
            pipe_to_index[edge] = idx

        for i, cycle in enumerate(cycles):
            # convert cycle (list of nodes) into list of directed edges
            for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
                if graph.has_edge(u, v):      # aligned edge
                    edge_id = graph[u][v]['id']
                    e = pipe_to_index[edge_id]
                    self.loop_matrix_active[i, e] = +1
                elif graph.has_edge(v, u):    # opposite direction edge, but set it to +1 as all directions are fixed
                    edge_id = graph[v][u]['id']
                    e = pipe_to_index[edge_id]
                    self.loop_matrix_active[i, e] = +1
        
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

        # dict of pump positions per pipe and its curve coeff
        self.pump_coeff = np.zeros((len(self.pipes),3))
        for pump in self.pumps.values():           
            pipe_id = pump.pipe_id
            j = self.pipe_map[pipe_id]
            self.pump_coeff[j,:] = [pump.a, pump.b, pump.c]  # Indicate presence of pump in this pipe
            
        # Vector of pressure per HEX. Which will be mapped to the inlet pipe of the HEX.
        self.Kp_array = np.zeros(len(self.pipes))
        self.inv_Kv_array = np.zeros(len(self.pipes))
        
        for hex_obj in self.hexs.values():
           
            # Put pressure drop of HEX on the inlet pipe
            pipe_id, pipe_obj = next(iter(hex_obj.get_incoming_pipes().items()))
            j = self.pipe_map[pipe_id]
            self.Kp_array[j] = hex_obj.Kp_rho_dp
        
        for valve in self.valves.values():
            j = self.pipe_map[valve.pipe_id]
            self.inv_Kv_array[j] = 1/valve.Kv[0] # Not necessary but for reverse engineering Rutger's data I left it here.

    def update_valves(self, N : int):
        """
        Build a boolean mask for pipes that are connected to a heat exchanger with an open valve. 
        And update valve positions
        """
        active_graph = self.G.copy()
        for valve in self.valves.values():
            
            # Only update valve state if not already done this timestep (debug purpose)
            already_updated = getattr(valve, 'updated_at', -1) == N
            if not already_updated:
                valve.update_valve(N)
                valve.updated_at = N

            if valve.h[N] > 0:
                j = self.pipe_map[valve.pipe_id]
                self.inv_Kv_array[j] = 1/valve.Kv[N]
            else:
                if 'Overflow' in valve.valve_id:
                    pipe_id = valve.pipe_id
                else:
                    pipe_id = valve.pipe_id
                    base, _ = pipe_id.split(".", 1)
                    pipe_id = f"{base}.2"

                from_node = self.pipes[pipe_id]['from']
                to_node = self.pipes[pipe_id]['to']
                active_graph.remove_edge(from_node,to_node)

        # source_node = 'Node 1.1'
        # reachable = nx.descendants(active_graph, source_node) | {source_node}
        # unreachable = set(active_graph.nodes()) - reachable
        # active_graph.remove_nodes_from(unreachable)

        cycles = nx.simple_cycles(active_graph.to_undirected())
        cycle_nodes = {n for c in cycles for n in c}
        active_graph_clean = active_graph.subgraph(cycle_nodes).copy()

        active_mask = np.zeros(len(self.pipes), dtype=bool)
        edge_ids = [data["id"] for _, _, data in active_graph_clean.edges(data=True)]

        for edge_id in edge_ids:
            j = self.pipe_map[edge_id]
            active_mask[j] = True
        
        return active_graph_clean, active_mask

    def res(self, mflow) -> np.ndarray:
        """
        Residual vector for Newton-Raphson method

        F(mflow) = [continuity equations; loop head-loss equations] = 0
        """
        # Continuity and loop head-loss equations
        continuity = self.incidence_matrix_active @ mflow

        head_loss  = self.loop_matrix_active @ (self.friction_vector_active * np.abs(mflow)*mflow
                                        + self.pressure_elevation_vector_active)

        # Pump contribution (only in loop equations)
        pump_pressure_curve =  self.pump_coeff_active[:,0] * mflow**2 + \
                                            self.pump_coeff_active[:,1] * mflow + \
                                            self.pump_coeff_active[:,2]
        
        
        pump_term = self.loop_matrix_active @ pump_pressure_curve

        # Full residual vector
        F = np.concatenate([continuity, head_loss - pump_term])
        
        return F
    
    def jac(self, mflow) -> np.ndarray:
        """
        Jacobian matrix of the residual F(x) for Newton-Raphson method
        """       
        pump_curve_derivative = 2 * self.pump_coeff_active[:,0] * mflow + self.pump_coeff_active[:,1]
        pump_term_derivative = self.loop_matrix_active @ np.diag(pump_curve_derivative)

        # d/dmflow |mflow|*mflow = |mflow| + mflow * mflow / |mflow|
        safe_abs = np.where(np.abs(mflow) > 1e-10, np.abs(mflow), 1e-10)
        deriv = safe_abs + mflow * mflow / safe_abs
        # deriv = np.abs(mflow) + mflow * mflow/np.abs(mflow)


        J = np.vstack([self.incidence_matrix_active, 
                       self.loop_matrix_active @ np.diag(2 * self.friction_vector_active * deriv) - pump_term_derivative]) 
        J += 1e-8 * np.eye(*J.shape)  # regularization
        return J

    def set_mflow_network(self, N : int):

        """
        Impelement root finding function to solve for mass flows in the network.

        Steps:
        1. Set up initial guess for mass flows in all pipes
        2. Reduce the incidence matrix by removing the first row (reference node)
        3. Set up the residual vector F(Q) containing continuity equations and loop head-loss equations
        4. Set up the Jacobian matrix J
        5. Solve for update in mass flows dQ: J dQ = -F(Q)
        6. Update mass flows: Q = Q + dQ

        """
        pipe_ids = np.array(list(self.pipes.keys()))
        p = len(pipe_ids)

        # Initial guess for the mflows in the pipes
        mflow0 = np.ones(p)*0.1
        
        active_graph, active_mask = self.update_valves(N)
        self.build_loop_matrix(active_graph)
        
        # In case all valves are closed, mflow is simply zero so skip hydraulic calculation
        if self.loop_matrix_active.shape[0] == 0:
            return

        # Reduce and create active incidence matrix
        self.incidence_matrix_red = self.incidence_matrix[1:,:]
        incidence_matrix_active = self.incidence_matrix_red[:, active_mask]
        self.incidence_matrix_active = incidence_matrix_active[~np.all(incidence_matrix_active == 0, axis=1), :]  # Remove zero rows (disconnected nodes)

        # Adjust all vectors to active pipes
        self.pressure_elevation_vector_active = self.pressure_elevation_vector[active_mask]
        self.pump_coeff_active = self.pump_coeff[active_mask, :]
        mflow0_active = mflow0[active_mask]

        friction_vector = self.pressure_friction_vector + \
                self.Kp_array + \
                self.inv_Kv_array**2
        self.friction_vector_active = friction_vector[active_mask]
        
        # Solves system of non linear equations using scipy root function
        result = root(self.res, mflow0_active, jac = self.jac, method = 'hybr', tol = 1e-6)

        # Retry with progressively rounded friction vectors if initial solve failed        
        if result.success == False or (result.x > 0).all():
            solved = False
            for precision in range(5,0,-1):

                # friction_vector = self.pressure_friction_vector + \
                #                 self.Kp_array + \
                #                 np.round(self.inv_Kv_array**2,precision)
                # self.friction_vector_active = friction_vector[active_mask]

                self.friction_vector_active = np.round(friction_vector,precision)[active_mask] 
                
                # Solves system of non linear equations using scipy root function
                result = root(self.res, mflow0_active, jac = self.jac, method = 'hybr', tol = 1e-6)

                # Try other root solver
                lm = False
                if result.success == False or (result.x < 0).all():
                    result = root(self.res, mflow0_active, jac = self.jac, method = 'lm', tol = 1e-6)
                    lm = True

                if result.success == True and (result.x > 0).all():
                    solved = True
                    if lm:
                        print(f'Used Levenbergh Marquardt at N = {N} and with precision {precision}')
                    else:
                        if precision < 5:
                            print(f'Used Hybr method at N = {N} and with precision {precision}')


                    break
            
            # In case root solving didn't work.
            if not solved: 
                for precision in range(5,0,-1): 
                    # friction_vector = self.pressure_friction_vector + \
                    #                 self.Kp_array + \
                    #                 np.round(self.inv_Kv_array**2,precision)
                    # self.friction_vector_active = friction_vector[active_mask]
                    self.friction_vector_active = np.round(friction_vector,precision)[active_mask] 


                    result = minimize(
                        fun=lambda x: np.sum(self.res(x)**2),
                        x0=mflow0_active,
                        jac=lambda x: 2 * self.jac(x).T @ self.res(x),  # chain rule: 2 * J^T * F
                        method='L-BFGS-B',
                        tol=1e-10
                    )

                    if result.success == True and (result.x > 0).all():
                        print(f'timestep {N} used minimization')
                        break

        # Extract results
        mflow_active = result.x

        # Save state of network for debugging if hydraulics break
        if not result.success or not (mflow_active > 0).all():
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       
                dis_steps = self.valves['Overflow valve'].steps
                Kvleak_bool = self.valves['Overflow valve'].Kvleak_bool

                c = int((self.pumps['Pump 1'].c)/1000)
                b = self.pumps['Pump 1'].b

                if b == 0:
                    pump_type = 'constant'
                else:
                    pump_type = 'curve'
                file_name = f"{self.net_id}_N={N}_Kvleak={Kvleak_bool}_hsteps={dis_steps}_pump={c}kPa_{pump_type}.pkl"

                with open(os.path.join(base_dir, 'debug', file_name), 'wb') as f:
                    pickle.dump(self.__dict__, f)

        if not result.success:   
            raise RuntimeError(f"The root finder did not converge at timestep = {N}, message: {result.message}")
        
        elif not (mflow_active > 0).all():
            raise RuntimeError(f'Newton-Raphson converges to a mass flow with negative values at timestep {N}')       

        self.mflow_all[active_mask,N] = mflow_active
        # Assign flows to the pipes for timestep N
        for j, pipe_id in enumerate(pipe_ids[active_mask]):
            pipe = self.pipes[pipe_id]['pipe_instance']
            pipe.set_mflow(mflow_active[j],N)
            pipe.save_dp_friction(N)

    def set_T_network(self, T_ambt : float, N : int, T_in : float, no_cap = False):
            
            """"
            Set temperature in the network.
            """

            # List of pipes for which the bnode method is performed.
            self.pipes_finished = []

            # Done by hand as no inflow pipe connected to node, gets Pipe 1. 
            first_pipe = self.pipes['Pipe 1.1']['pipe_instance']
            first_node = self.nodes['Node 1.1'] 

            # set_T doesn't work as there is no mass inflow, so set it directly.
            first_node.T[N] = T_in 
            first_pipe.set_T_in(first_node.T[N], N) # Set inlet temperature

            first_pipe.bnode_method_fast(T_ambt, N, no_cap = no_cap)

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

            # if pipe_id != 'Pump 1':

            pipe.bnode_method_fast(T_ambt, N, no_cap = no_cap)
            self.pipes_finished.append(pipe_id)
            
            next_node_id = self.pipes[pipe_id]['to']
            next_node = self.nodes[next_node_id]

            if all(pipes in self.pipes_finished for pipes in list(next_node.pipes_in.keys())):

                if next_node_id == 'Node 1.1':
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
















