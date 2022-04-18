import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

class ShowPC():
    def __init__(self, points):
        all = points.squeeze(0).cpu().detach().numpy()
        x = [k[0] for k in all]
        y = [k[1] for k in all]
        z = [k[2] for k in all]
        fig=plt.figure(dpi=120)
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,z,c='b',marker='o',s=10,linewidth=0,alpha=1,cmap='spectral')
        # ax.axis('scaled')          
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()