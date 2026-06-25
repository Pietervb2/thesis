"""
Recreates the linear valve characteristic plot using the exact
definitions from valve.py linear_valve() method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Constants directly from valve.py linear_valve()
# ---------------------------------------------------------------------------
Kvs    = 1.0              # normalised to 1 for the plot
Kv0    = 0.2         # = 0.04
Kvleak = Kvs / 100  # = 0.0005
h_star = 0.1

# Kvr: Kv value at h* on the linear line (top of leakage transition)
Kvr = Kv0 + h_star * (Kvs - Kv0)   # from valve.py: Kv0 + h_star*(Kvs - Kv0)

# ---------------------------------------------------------------------------
# Compute Kv(h) over [0, 1]  — vectorised version of linear_valve()
# ---------------------------------------------------------------------------
h = np.linspace(0, 1, 2000)

Kv = np.where(
    h < h_star,
    Kvleak + h * (Kvr - Kvleak) / h_star,   # leakage region (Kvleak_bool=True)
    Kv0 + h * (Kvs - Kv0)                   # linear region
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4.5))

ax.plot(h, Kv, color='black', linewidth=1.5)

# --- Dotted reference lines ---
# Kvr at h*
ax.plot([0, h_star], [Kvr, Kvr],   color='black', linewidth=0.8, linestyle='dotted')
ax.plot([h_star, h_star], [0, Kvr], color='black', linewidth=0.8, linestyle='dotted')
# Kv0 horizontal
ax.plot([0, h_star], [Kv0, Kv0],   color='black', linewidth=0.8, linestyle='dotted')

# Transition point dot at (h*, Kvr)
ax.plot(h_star, Kvr, 'o', color='black', markersize=4, zorder=5)

# --- Y-axis labels (manual, left of axis) ---
offset = -0.04
ax.text(offset, Kvs,    r'$K_{vs}$',     ha='right', va='center', fontsize=11)
ax.text(offset, Kvr,    r'$K_w$',        ha='right', va='center', fontsize=11)
ax.text(offset, Kv0,    r'$K_{v0}$',     ha='right', va='center', fontsize=11)
ax.text(offset, Kvleak, r'$K_{v,leak}$', ha='right', va='center', fontsize=11)

# --- X-axis label for h* ---
ax.text(h_star, -0.028, r'$h^*$', ha='center', va='top', fontsize=11)
ax.text(1.02,   -0.028, r'$1$',   ha='center', va='top', fontsize=11)
ax.text(-0.005, -0.028, r'$0$',   ha='center', va='top', fontsize=11)

# --- Axes formatting ---
ax.set_xlim(-0.01, 1.05)
ax.set_ylim(-0.03, Kvs * 1.12)

ax.set_xlabel(r'$h_{lift}$ (-)', fontsize=11, labelpad=10)
ax.set_ylabel(r'$K_v\ (\mathrm{m^3/h \cdot bar^{0.5}})$', fontsize=11)

# Clean thesis style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f'base_dir: {base_dir}')
save_path = os.path.join(base_dir, 'generate_valve_plot.png')

plt.savefig(save_path, dpi=150, bbox_inches='tight')
print('Done')