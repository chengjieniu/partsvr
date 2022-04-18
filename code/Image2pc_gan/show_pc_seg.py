import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

class ShowPC_SEG():
    def __init__(self, points1, points2, points3, points4):
        all1 = points1.squeeze(0).cpu().detach().numpy()
        all2 = points2.squeeze(0).cpu().detach().numpy()
        all3 = points3.squeeze(0).cpu().detach().numpy()
        all4 = points4.squeeze(0).cpu().detach().numpy()
        x1 = [k[0] for k in all1]
        y1 = [k[1] for k in all1]
        z1 = [k[2] for k in all1]

        x2 = [k[0] for k in all2]
        y2 = [k[1] for k in all2]
        z2 = [k[2] for k in all2]

        x3 = [k[0] for k in all3]
        y3 = [k[1] for k in all3]
        z3 = [k[2] for k in all3]

        x4 = [k[0] for k in all4]
        y4 = [k[1] for k in all4]
        z4 = [k[2] for k in all4]

        fig=plt.figure(dpi=120)
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(x1,y1,z1,c='b',marker='o',s=2,linewidth=0,alpha=1,cmap='spectral')
        ax.scatter(x2,y2,z2,c='r',marker='o',s=2,linewidth=0,alpha=1,cmap='spectral')
        ax.scatter(x3,y3,z3,c='c',marker='o',s=2,linewidth=0,alpha=1,cmap='spectral')
        ax.scatter(x4,y4,z4,c='g',marker='o',s=2,linewidth=0,alpha=1,cmap='spectral')

        elev = -33. 
        azim = 33
        ax.view_init(elev, azim)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴

        # ax.axis('scaled')          
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        # plt.axis('off')  #去掉坐标轴plt.axis('off')  #去掉坐标轴