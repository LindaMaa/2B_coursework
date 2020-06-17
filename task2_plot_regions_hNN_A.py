#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import matplotlib.pyplot as plt
import task2_hNN_A 

def task2_plot_regions_hNN_A():
    
    # set up the grid for plotting
    x_range = np.arange(1,4, 0.005)
    y_range = np.arange(1,4,0.005)
    xs, ys = np.meshgrid(x_range, y_range)
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    grid = np.array([[xs_flat[k], ys_flat[k]] for k in range(len(xs_flat))])


    # classify points using task2_hNN_A
    dataGrid = task2_hNN_A.task2_hNN_A(grid)
    dataGrid = dataGrid.reshape((x_range.shape[0], y_range.shape[0]))
    
    fig = plt.figure()
    plt.xticks(np.arange(0, 4, 0.5), fontsize=8)
    plt.yticks(np.arange(0, 4, 0.5),fontsize=8)
    plt.title('Decision Regions hNN_A')

    # plot & save the figure
    contourPlot = plt.contourf(x_range, y_range, dataGrid, cmap='Reds')
    proxy = np.array([plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in contourPlot.collections])
    # add a legend
    proxy = [proxy[0], proxy[-1]]
    plt.legend(proxy, ['Class 0', 'Class 1'])
    plt.show()
    plt.draw()
    fig.savefig('t2_regions_hNN_A.pdf')
    
if __name__ == "__main__":
    task2_plot_regions_hNN_A()
