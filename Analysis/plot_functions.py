import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from error_functions import calculate_prediction

## Plot functions


def plot_prediction(t, V, data_indexes, kernel_times, delay_indexes, image_indexes, 
                     input_to_image, kernel_to_input, h0, h1, h2, ims, ims2):
   
    prediction = calculate_prediction(data_indexes, kernel_times, delay_indexes, image_indexes, 
                                       input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
    # Plot data 
    plt.plot(t,V[data_indexes], '-*', label='Data')
    plt.hold('on')
    # Plot prediction
    plt.plot(t,prediction, '-*', label='Prediction')
    plt.legend()
    
    

def plot_mutliplot(N,ims):
    n = int(np.sqrt(N))
    print n
    number = n * n
    print number
    for i in range(number):
        plt.subplot(n,n,i + 1)
        plt.imshow(ims[i,:,:], interpolation='nearest')
        plt.colorbar()

def plot_mutliplot_bilinear(N,ims, colorbar=True, symmetric=0, aux1=False, aux2=False):
    """
    Plots a series of 2D images as imshow functions. 
    
    ---
    Parameters
    N: This is the number of actual images that are going to be ploted. This function will round to the closest perfect square 
    number n and then it will make an array of n x n figures.
    
    colorbar: True if you want the colorbars and False if the oposite
    
    symmetric:  0 sets a symmetric ticks with maximum of abs(ims) as vmax and the negative of that value a vmin
                1 set the maximum of ims as vmax and minimum as min
                2 no values, default of imshow
    """
    
    # Set the color map
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),
    
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
    
         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }
    
    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    
    # Set the maximums and minimums 
    if symmetric == 0:
        vmax = np.max((np.abs(np.min(ims)), np.max(ims)))
        vmin = - vmax
    elif symmetric == 1:
        vmax = np.max(ims)
        vmin = np.min(ims)
    elif symmetric == 2:
        vmin = aux1
        vmax = aux2
    else:
        vmax = None
        vmin = None
    
    n = int(np.sqrt(N))
    number = n * n

    if colorbar:
        figure, axes = plt.subplots(nrows=n, ncols=n)

        for i, ax in enumerate(axes.flat):
            # The vmin and vmax arguments specify the color limits
            im = ax.imshow(ims[i,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)

        # Make an axis for the colorbar on the right side
        cax = figure.add_axes([0.9, 0.1, 0.03, 0.8])
        figure.colorbar(im, cax=cax)
    else:
         for i in range(number):
             plt.subplot(n,n,i + 1)
             plt.imshow(ims[i,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)


##to make comparisons easier, we are plotting the RF and the time-trace for 1 cell for 1 pixel.
#def plot_mutliplot_compare(N,ims, colorbar=True, symmetric=0, aux1=False, aux2=False):
#    """
#    Plots a series of 2D images as imshow functions. 
#    
#    ---
#    Parameters
#    
#    colorbar: True if you want the colorbars and False if the oposite
#    
#    symmetric:  0 sets a symmetric ticks with maximum of abs(ims) as vmax and the negative of that value a vmin
#                1 set the maximum of ims as vmax and minimum as min
#                2 no values, default of imshow
# 
#    """
#    # Set the color map
#    cdict1 = {'red':   ((0.0, 0.0, 0.0),
#                   (0.5, 0.0, 0.1),
#                   (1.0, 1.0, 1.0)),
#    
#         'green': ((0.0, 0.0, 0.0),
#                   (1.0, 0.0, 0.0)),
#    
#         'blue':  ((0.0, 0.0, 1.0),
#                   (0.5, 0.1, 0.0),
#                   (1.0, 0.0, 0.0))
#        }
#    
#    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
#    
#    # Set the maximums and minimums 
#    if symmetric == 0:
#        vmax = np.max((np.abs(np.min(ims)), np.max(ims)))
#        vmin = - vmax
#    elif symmetric == 1:
#        vmax = np.max(ims)
#        vmin = np.min(ims)
#    elif symmetric == 2:
#        vmin = aux1
#        vmax = aux2
#    else:
#        vmax = None
#        vmin = None
#
#    #plot RF
#   if colorbar:
#        figure, axes = plt.subplot(52)
#        
#        for i, ax in enumerate(axes.flat):
#            # The vmin and vmax arguments specify the color limits
#            im = ax.imshow(ims[i,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
#
#        # Make an axis for the colorbar on the right side
#        cax = figure.add_axes([0.9, 0.1, 0.03, 0.8])
#        figure.colorbar(im, cax=cax)        
#        
#        
#    plt.imshow()
#    
#    #plot time traces
#    plt.subplot(522)
#    plt.imshow()
#
