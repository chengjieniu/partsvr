import numpy as np
import show3d_balls
import torch

# COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([248, 250, 132]), np.array([237, 186, 145]),
#           np.array([192, 144, 145]), np.array([158, 218, 73])]
# most chairs 
COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([192, 144, 145]),np.array([248, 250, 132]),
          np.array([192, 144, 145]), np.array([248, 250, 132])]

# COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([205, 92, 92]),np.array([0, 139, 139]),
#           np.array([0, 0, 0]),np.array([0, 0, 0])]

NUM_POINTS = 2000
NUM_PARTS = 4
def show_3d_point_clouds(shapes, is_missing_part):
    colors = np.zeros_like(shapes)
    for p in range(NUM_PARTS):
        colors[NUM_POINTS * p: NUM_POINTS * (p + 1), :] = COLORS[p]

    # fix orientation
    # shapes[:, 1] *= -1
    # shapes = shapes[:, [2, 1, 0]]

    if is_missing_part:
        shapes = shapes[:NUM_POINTS * (NUM_PARTS - 1)]
        colors = colors[:NUM_POINTS * (NUM_PARTS - 1)]
    # show3d_balls.showpoints(shapes, c_gt=colors, ballradius=8, normalizecolor=False, background=[255, 255, 255])
    frame = show3d_balls.showpoints_frame(shapes, c_gt=colors, ballradius=8, normalizecolor=False, background=[255, 255, 255])
    return frame

# if __name__ == "__main__":
#     shapes = torch.randn([8000, 3])
#     show_3d_point_clouds(shapes, False)