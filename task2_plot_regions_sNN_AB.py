#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import matplotlib.pyplot as plt
import task2_sNN_AB
import numpy as np

def task2_plot_regions_sNN_AB():

    # set up the grid for plotting
    x_range = np.arange(-1.5, 7.5, 0.01)
    y_range = np.arange(-2, 7, 0.01)
    xs, ys = np.meshgrid(x_range, y_range)
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    grid = np.array([[xs_flat[k], ys_flat[k]] for k in range(len(xs_flat))])

    # classify points using task2_sNN_AB
    dataGrid = task2_sNN_AB.task2_sNN_AB(grid)
    dataGrid = dataGrid.reshape((x_range.shape[0], y_range.shape[0]))

    fig = plt.figure()
    plt.xticks(np.arange(-2, 8, 1), fontsize=8)
    plt.yticks(np.arange(-3, 7, 1), fontsize=8)
    plt.title('Decision Regions sNN_AB')

    # plot & save the figure
    contourPlot = plt.contourf(x_range, y_range, dataGrid, cmap='Greens')
    # legend
    proxy = np.array([plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in contourPlot.collections])
    proxy = [proxy[0], proxy[-1]]
    plt.legend(proxy, ['Class 0', 'Class 1'])
    plt.show()
    plt.draw()
    fig.savefig('t2_regions_sNN_AB.pdf')


if __name__ == "__main__":
    task2_plot_regions_sNN_AB()

