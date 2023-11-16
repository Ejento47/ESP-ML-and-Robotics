import numpy as np
filename = 'trained_q_table.npy'
data = np.load(filename)
print(data)
print(data.shape)
# if data contains values > 0
# then it is a trained q table
# else it is an empty q table
if np.any(data > 0):
    print('trained q table')