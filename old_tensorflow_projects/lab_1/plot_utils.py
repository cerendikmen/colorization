import numpy as np;
from src.lab_1 import math_utils, neural_networks
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.mlab import griddata;
def plot_decision_boundary(set_data, set_target,W,plot_legend):
    x = (int(np.min(set_data[:,0])-1),int(np.max(set_data[:,0])+1))
    y = (math_utils.findYFromNullSpace(W,x))
    r = set_data;
    t = set_target;
    colArr = [];
    for v in t:
        if(v == 1):
            colArr.append("r");
        else: colArr.append("b");
    plt.plot(x,y,"-",label = plot_legend);
def surface_plot_from_xyz(x,y,z,color_z_min, color_z_max, color_bar=True):
    root_x = np.sqrt(len(x));
    root_y = np.sqrt(len(y));

    X = np.reshape(x, [root_x,root_x])
    Y = np.reshape(y, [root_y,root_y])
    Z = np.reshape(z, [root_x,root_x])
    
    #plt.pcolormesh(X,Y,Z)
    #plt.show()
    z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()

    if(color_z_min != None and color_z_max != None ) :
        z_min, z_max = color_z_min, color_z_max;
    mesh = plt.pcolormesh(X, Y, Z, cmap='RdBu', vmin=z_min, vmax=z_max);
    # set the limits of the plot to the limits of the data
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    if(color_bar):
        plt.colorbar()
    return mesh;
