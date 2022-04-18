import torch
import numpy as np
from show_pc import ShowPC

import matplotlib.pyplot as plt

# change point from [] to [-0.5, 0.5]
def PC_Normalize( point): #point N*3
    all = point.squeeze(0).cpu().detach().numpy()
    x = [k[0] for k in all]
    y = [k[1] for k in all]
    z = [k[2] for k in all]

    x_one = max(x) - min(x)
    x_after = (x - min(x))/ x_one - 0.5

    y_one = max(y) - min(y)
    y_after = (y - min(y)) / y_one - 0.5

    z_one = max(z) - min(z)
    z_after = (z - min(z)) / z_one - 0.5
    point = torch.cat([torch.from_numpy(x_after).unsqueeze(1), torch.from_numpy(y_after).unsqueeze(1), torch.from_numpy(z_after).unsqueeze(1)], 1)
    return point

    