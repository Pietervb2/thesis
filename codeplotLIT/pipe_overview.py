"""
Recreates the pipe cross-section figure (panel a only) from pijpdoorsnede.jpg,
removing the casing layer, casing label, r_d arrow, and panel (b).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_aspect('equal')

# ---------------------------------------------------------------------------
# Radii (normalised, no casing so r_d removed)
# ---------------------------------------------------------------------------
r_a = 0.30   # inner pipe wall (fluid boundary)
r_b = 0.38   # outer pipe wall
r_c = 0.70   # outer insulation

# ---------------------------------------------------------------------------
# Colors matching the original figure
# ---------------------------------------------------------------------------
color_fluid      = 'white'
color_pipe       = '#4a4a4a'       # dark grey pipe wall
color_insulation = '#b0b0b0'       # light grey insulation

# ---------------------------------------------------------------------------
# Draw layers (outermost first so inner layers paint over)
# ---------------------------------------------------------------------------

# Insulation
insulation = plt.Circle((0, 0), r_c, color=color_insulation, zorder=1)
ax.add_patch(insulation)

# Pipe wall
pipe = plt.Circle((0, 0), r_b, color=color_pipe, zorder=2)
ax.add_patch(pipe)

# Fluid interior
fluid = plt.Circle((0, 0), r_a, color=color_fluid, zorder=3)
ax.add_patch(fluid)

# Thin black outlines
for r, zorder in [(r_c, 4), (r_b, 5), (r_a, 6)]:
    circle = plt.Circle((0, 0), r, fill=False,
                         edgecolor='black', linewidth=0.8, zorder=zorder)
    ax.add_patch(circle)

# ---------------------------------------------------------------------------
# Radius arrows: r_a, r_b, r_c  (diagonal, upper-right quadrant)
# ---------------------------------------------------------------------------
angle_ra = np.radians(0)   # arrow direction
angle_rb = angle_ra + np.pi/4
angle_rc = np.radians(90)
def draw_radius_arrow(ax, r, label, angle, label_offset=(0.03, 0.03)):
    x_end = r * np.cos(angle)
    y_end = r * np.sin(angle)
    ax.annotate('', xy=(x_end, y_end), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
    ax.text(x_end + label_offset[0], y_end + label_offset[1],
            label, fontsize=10, ha='left', va='bottom')

draw_radius_arrow(ax, r_a, r'$r_a$', angle_ra, label_offset=(-0.1,  -0.08))
draw_radius_arrow(ax, r_b, r'$r_b$', angle_rb, label_offset=(0.02,  0.01))
draw_radius_arrow(ax, r_c, r'$r_c$', angle_rc, label_offset=(0.02,  0.01))

# ---------------------------------------------------------------------------
# Labels with leader lines (left side, matching original)
# ---------------------------------------------------------------------------
label_x = -1.15

def leader_label(ax, label, y_text, r_target, angle_leader=np.radians(200)):
    x_circle = r_target * np.cos(angle_leader)
    y_circle = r_target * np.sin(angle_leader)
    ax.annotate('', xy=(x_circle, y_circle),
                xytext=(label_x + 0.35, y_text),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.8))
    ax.text(label_x + 0.30, y_text, label,
            fontsize=10, ha='right', va='center')

leader_label(ax, 'Pipe',        y_text= 0, r_target=(r_a + r_b) / 2)
leader_label(ax, 'Insulation',  y_text=-0.20, r_target=(r_b + r_c) / 2)

# ---------------------------------------------------------------------------
# Coordinate axes (top-left corner)
# ---------------------------------------------------------------------------
ax_origin = (-0.88, 0.78)
arrow_len  = 0.18
ax.annotate('', xy=(ax_origin[0] + arrow_len, ax_origin[1]),
            xytext=ax_origin,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
ax.annotate('', xy=(ax_origin[0], ax_origin[1] + arrow_len),
            xytext=ax_origin,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
ax.text(ax_origin[0] + arrow_len + 0.04, ax_origin[1],     'z', fontsize=10, va='center')
ax.text(ax_origin[0],                    ax_origin[1] + arrow_len + 0.05, 'y', fontsize=10, ha='center')

# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------
margin = 1.05
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.axis('off')

plt.tight_layout()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thesis_fig_folder = os.path.join(base_dir, 'Thesis report', 'figures_thesis')
loc = os.path.join(thesis_fig_folder, 'pipe_cross_section.png')
plt.savefig(loc, dpi=150, bbox_inches='tight')
print('Done')