import scipy.io as scio
from scipy.io import loadmat
import numpy
import matplotlib.pyplot as plt

purity_data = loadmat('/home/ncj/Desktop/GMP2020/Image2pc_semantic/snap_shots/snapshots_2019-12-13_13-56-19/plot_purity.mat')
plt.title('Result Analysis')
total_iter = purity_data['iter'][0].max()
plot_x = [x for x in range(total_iter)]
# plot_loss = [None for x in range(total_iter)]
# plot_rloss = [None for x in range(total_iter)]
# plot_geloss = [None for x in range(total_iter)]
# plot_dloss = [None for x in range(total_iter)]
# dyn_plot = DynamicPlot(title='Training loss over epochs (I2PC_part)', xdata=plot_x, \
#             ydata={'purity_loss':purity_data['purity_loss'], 'total_loss': purity_data[ 'total_loss'] })
# iter_id = 0

plot_loss = purity_data[ 'loss'].squeeze()
plot_purity = purity_data['purity_loss'].squeeze()
plot_recons = purity_data['total_loss'].squeeze()
# plt.plot(plot_x, plot_loss, color='green', label='loss' )
plt.plot(plot_x, plot_purity, color='red', label='plot_purity' )
# plt.plot(plot_x, plot_recons, color='skyblue', label='plot_recons' )
plt.legend()

plt.xlabel('iterator')
plt.ylabel('loss')
plt.show()
# plot_rloss[iter_id] = total_loss.item()

# plot_geloss[iter_id] = gen_loss.item()*2
# plot_dloss[iter_id] = dis_loss.item()*1
# max_loss = max(max_loss,loss.item())

