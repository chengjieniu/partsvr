import numpy as np
import show3d_balls
import torch

# COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([248, 250, 132]), np.array([237, 186, 145]),
#           np.array([192, 144, 145]), np.array([158, 218, 73])]
# COLORS = [np.array([163, 254, 170]), np.array([206, 178, 254]), np.array([192, 144, 145]),np.array([248, 250, 132]),
#           np.array([192, 144, 145]), np.array([248, 250, 132])]
COLORS = [np.array([192, 192, 192])]
NUM_POINTS = 8000
NUM_PARTS = 1
def show_3d_point_clouds(shapes, is_missing_part):
    # for gt
    if len(shapes)>8000:
        shapes = shapes[0:7999,:]
    # for i in range(8000):
    #     for j in range(3):
            # if j == 1:
            #     if abs(shapes[i, j]) >= 0.2: 
            #         shapes[i,:] = [0 ,0, 0]

            # if abs(shapes[i, j]) >= 0.35: 
            #     shapes[i,:] = [0 ,0, 0]

    colors = np.zeros_like(shapes)
    for p in range(NUM_PARTS):
        colors[NUM_POINTS * p:NUM_POINTS * (p + 1), :] = COLORS[p]

    # fix orientation
    # shapes[:, 1] *= -1
    # shapes = shapes[:, [1, 2, 0]]

    #chair basic unsuper
    # shapes[:, 1] *= -1
    # shapes = shapes[:, [1, 0 ,2]]

    # car basic unsuper
    # shapes = shapes[:, [1, 2 ,0]]

    # airplan basic unsuper
    # shapes[:, 1] *= 1
    # shapes = shapes[:, [1, 0, 2]]

    # shapes[:, 1] *= 1
    # for chair render2 unsuper
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