import numpy as np
import matplotlib.pyplot as plt
import os
# Define the range
x= np.linspace(0,1,101)

# Define the steep and less steep slopes
steep_slope = 2.7
less_steep_slope = 0.7

# Linear functions
y1 = steep_slope * x + 0.1
y2 = y1[10] + less_steep_slope * (x - 0.1) 

# Plot the steep part
plt.plot(x[:11], y1[:11], label='Steep Linear', color = 'black')

# Plot the less steep part (dotted in 0 to 0.1)
plt.plot(x[11:], y2[11:], label='Less Steep (0-0.1)', color = 'black')

# # Plot the less steep part (solid in 0.1 to 1)
plt.plot(x[:11], y2[:11], 'k--', color = 'black')

# Dotted lines at the change of gradient
plt.axvline(x=0.1, color='gray', linestyle=':', ymax=y1[10])
plt.axhline(y=y1[10], color='gray', linestyle=':', xmax=0.1)

font = 14
# Remove intermediate valve ticks between 0 and 1 on both axes
plt.xticks([0, 0.1, 1], ['$0$', '$h^*$', '$1$'], fontsize=font)
plt.yticks([0, y1[0], y1[10] - less_steep_slope*0.1, y1[10], 1], ['$0$', '$K_{v,\mathrm{leak}}$' , '$K_{v0}$','$K_{vr}$', '$K_{vs}$'], fontsize=font)

# Axis limits and labels
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.ylabel('$K_{v} (m^3/h/\sqrt{bar})$', fontsize=font)
plt.xlabel('$h_{lift}$ (-)', fontsize=font)
plt.tight_layout()
# plt.show()

LitSurFig = 'Literature Survey - DCSC template/figuresLIT'
current_folder = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_folder,LitSurFig, 'Kvs_plot.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')  # High quality with tight layout