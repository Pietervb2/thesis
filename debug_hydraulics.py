from baseclasses.network import Network
import pickle
import os


net_id = 'Profile 1'
Kvleak_bool = True
dis_steps = 125
c = 50
pump_type = 'constant'

file_name = f"{net_id}_Kvleak={Kvleak_bool}_hsteps={dis_steps}_pump={c}kPa_{pump_type}.pkl"
base_dir = os.path.dirname(__file__)
file = os.path.join(base_dir, 'debug', file_name)

with open(file, 'rb') as f:
    state = pickle.load(f)

net = Network.__new__(Network)  # create instance without calling __init__
net.__dict__.update(state)

print(net.loop_matrix_active.shape[0])