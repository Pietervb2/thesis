from .node import Node

import numpy as np

class HeatExchanger(Node):

    def __init__(self,
                 x,
                 y,
                 z,
                 hex_id: str,
                 hex_data: list,
                 consumer: object,
                 C_h: float = None,
                 C_wall: float = None,
                 C_c: float = None):
        """
        Args:
            - U: Overall heat transmission coefficient [W/m2K]
            - As: Total heat transfer area [m2]
            - Kp_rho_dp: Pressure loss coefficient [kg/m2]
            - Kvs: Hydraulic conductivity of fully open valve [m3/h/bar^0.5]
            - Kp: Proportional gain of the PI controller [-]
            - Ki: Integral gain of the PI controller [-]
            - C_h: Thermal mass of hot-side fluid [J/K]. If None, estimated from As.
            - C_wall: Thermal mass of plate wall [J/K]. If None, estimated from As.
            - C_c: Thermal mass of cold-side fluid [J/K]. If None, estimated from As.

        Three-node dynamic model
        ------------------------
        The HEX is represented as three coupled lumped nodes:

            Hot fluid (T_h) <--> Wall (T_wall) <--> Cold fluid (T_c)

        Each node satisfies an ODE:

            C_h    * dT_h/dt    =  m_h*cp*(T_h_in - T_h)   - UA_h*(T_h - T_wall)
            C_wall * dT_wall/dt =  UA_h*(T_h - T_wall)      - UA_c*(T_wall - T_c)
            C_c    * dT_c/dt    =  m_c*cp*(T_c_in - T_c)   + UA_c*(T_wall - T_c)

        UA_h and UA_c are the individual side conductances, scaled with
        Dittus-Boelter (mflow^0.8) from their design values.

        Thermal mass defaults
        ---------------------
        Estimated from As assuming:
            - Fluid gap per side: 2 mm  -> V_fluid = As * 0.002 / 2
            - Plate thickness:    0.5 mm -> m_wall  = As * 0.0005 * rho_steel
        These are calibration parameters; override with measured values when available.
        """
        super().__init__(x, y, z, hex_id)

        self.U          = hex_data[0]
        self.As         = hex_data[1]
        self.Kp_rho_dp  = hex_data[2] * 1000   # [Pa / (kg/s)^2]
        self.Kvs        = hex_data[3]
        self.Kp         = hex_data[4]
        self.Ki         = hex_data[5]

        self.consumer = consumer

        # Design flow rates for UA scaling (Dittus-Boelter)
        self.mflow_h_des = 0.247   # kg/s  (31e3 / (30 * 4181))
        self.mflow_c_des = 0.15    # kg/s

        # Thermal masses
        rho_w, c_w   = 1000.0, 4186.0   # water
        rho_s, c_s   = 7900.0,  500.0   # stainless steel

        gap    = 0.002    # m, fluid channel gap per side
        s_wall = 5e-4     # m, plate thickness

        V_fluid = self.As * gap / 2       # m^3 per fluid side

        self.C_h    = C_h    if C_h    is not None else rho_w * c_w * V_fluid
        self.C_wall = C_wall if C_wall is not None else rho_s * c_s * self.As * s_wall
        self.C_c    = C_c    if C_c    is not None else rho_w * c_w * V_fluid

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_node(self, num_steps: int, T_init: float, dt: float) -> None:
        """
        Overrides Node.initialize_node.
        Initialises fluid and wall temperature state arrays.
        """
        super().initialize_node(num_steps, T_init, dt)
        self.consumer.initialize_consumer(num_steps, dt)

        self.dt = dt

        # Three-node state arrays
        T_c_init = self.consumer.Tc_in[0]

        self.T_h    = np.ones(num_steps) * T_init
        self.T_wall = np.ones(num_steps) * ((T_init + T_c_init) / 2)
        self.T_c    = np.ones(num_steps) * T_c_init

        # Store initial values for use at N=0
        self._T_h_init    = T_init
        self._T_wall_init = (T_init + T_c_init) / 2
        self._T_c_init    = T_c_init

    # ------------------------------------------------------------------
    # Node temperature update (called by network solver each timestep)
    # ------------------------------------------------------------------

    def set_T(self, N: int) -> None:
        """
        Overrides Node.set_T.
        Integrates the three-node ODEs and propagates the hot-side
        outlet temperature to downstream pipes.
        """
        Tc_out, Th_out = self._three_node_step(N)

        self.T[N] = Th_out

        for _, pipe_out in self.pipes_out.items():
            pipe_out.set_T_in(self.T[N], N)

    # ------------------------------------------------------------------
    # Three-node dynamic model
    # ------------------------------------------------------------------

    def _three_node_step(self, N: int):
        """
        Advances the three-node model by one timestep using implicit Euler.

        Forward Euler is unconditionally unstable for this system when
        dt > 2*tau_min, where tau_min = C_wall/(UA_h+UA_c) ~ 0.12 s at
        design flow — far below the typical simulation dt of 1 s.

        Implicit Euler discretises each ODE as:
            C * (T[N] - T[N-1]) / dt = f(T[N])

        which rearranges to the 3x3 linear system A*x = b, solved exactly
        each timestep. This is unconditionally stable regardless of dt.

        System matrix A (rows: hot fluid, wall, cold fluid):

            | C_h/dt + m_h*cp + UA_h  -UA_h                    0              |
            | -UA_h                    C_w/dt + UA_h + UA_c    -UA_c           |
            | 0                       -UA_c                    C_c/dt+m_c*cp+UA_c |

        RHS vector b:
            [ C_h/dt * T_h_prev    + m_h*cp * T_h_in  ]
            [ C_w/dt * T_wall_prev                     ]
            [ C_c/dt * T_c_prev    + m_c*cp * T_c_in  ]

        Returns
        -------
        Tc_out : float  Cold-side outlet temperature [°C]
        Th_out : float  Hot-side outlet temperature  [°C]
        """
        # --- Inlet conditions -------------------------------------------
        pipe    = self.pipes_in[f'Pipe {self.node_id.split()[-1]}.3']
        Th_in   = pipe.T[N]
        mflow_h = pipe.get_mflow(N)

        Tc_in   = self.consumer.Tc_in[N]
        mflow_c = self.consumer.mflow[N]

        c_p = pipe.c_water

        # --- Previous states --------------------------------------------
        if N == 0:
            T_h_prev    = self._T_h_init
            T_wall_prev = self._T_wall_init
            T_c_prev    = self._T_c_init
        else:
            T_h_prev    = self.T_h[N-1]
            T_wall_prev = self.T_wall[N-1]
            T_c_prev    = self.T_c[N-1]

        # --- Flow-dependent side conductances ---------------------------
        UA_h, UA_c = self.compute_UA_sides(mflow_h, mflow_c)

        # --- Build 3x3 implicit Euler system ----------------------------
        a_h  = self.C_h    / self.dt
        a_w  = self.C_wall / self.dt
        a_c  = self.C_c    / self.dt

        mh_cp = mflow_h * c_p
        mc_cp = mflow_c * c_p

        A = np.array([
            [a_h + mh_cp + UA_h,  -UA_h,                    0.0               ],
            [-UA_h,                a_w + UA_h + UA_c,       -UA_c              ],
            [0.0,                 -UA_c,                     a_c + mc_cp + UA_c]
        ])

        b = np.array([
            a_h * T_h_prev    + mh_cp * Th_in,
            a_w * T_wall_prev,
            a_c * T_c_prev    + mc_cp * Tc_in
        ])

        T_new = np.linalg.solve(A, b)

        self.T_h[N]    = T_new[0]
        self.T_wall[N] = T_new[1]
        self.T_c[N]    = T_new[2]

        Th_out = self.T_h[N]
        Tc_out = self.T_c[N]

        # --- Consumer bookkeeping ---------------------------------------
        Q_supply = mflow_c * c_p * max(0.0, Tc_out - Tc_in)
        self.consumer.Tc_out[N] = Tc_out
        self.consumer.Q_supply[N] = Q_supply

        return Tc_out, Th_out

    # ------------------------------------------------------------------
    # UA calculation
    # ------------------------------------------------------------------

    def compute_UA_sides(self, mflow_h: float, mflow_c: float, n: float = 0.8):
        """
        Individual side conductances scaled with Dittus-Boelter (h ~ mflow^0.8).

        Derivation:
            R_i = (1/UA_i_des) * (mflow / mflow_des)^{-n}
            UA_i = 1/R_i = UA_i_des * (mflow / mflow_des)^n

        Returns
        -------
        UA_h : float  Hot-side conductance  [W/K]
        UA_c : float  Cold-side conductance [W/K]
        """
        mflow_h = max(mflow_h, 1e-6)
        mflow_c = max(mflow_c, 1e-6)

        UA_des      = self.U * self.As
        UA_h_des    = 2 * UA_des   # split evenly between sides at design point
        UA_c_des    = 2 * UA_des

        UA_h = UA_h_des * (mflow_h / self.mflow_h_des) ** n
        UA_c = UA_c_des * (mflow_c / self.mflow_c_des) ** n

        return UA_h, UA_c

    def compute_UA(self, mflow_h: float, mflow_c: float, n: float = 0.8) -> float:
        """
        Combined overall UA [W/K] for reference or external use.
        Kept for backward compatibility; the dynamic model uses compute_UA_sides.
        """
        UA_h, UA_c = self.compute_UA_sides(mflow_h, mflow_c, n)
        return 1.0 / (1.0/UA_h + 1.0/UA_c)