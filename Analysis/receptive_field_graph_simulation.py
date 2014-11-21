import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from plot_functions import plot_mutliplot_bilinear

## Files
images_folder = './data/'
kernels_folder = './kernels/'
quality = '_15000_21_'
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'

# Save things or not
remove_axis = True # Remove axis if True 
show_plot = True # Show or not the plot

number_of_cells_to_plot = 2
list_cells = np.arange(1, number_of_cells_to_plot +1 )



for i, cell_number in enumerate(list_cells):
    
    plt.close('all') #close the plot for memory issues    
    
    cell_number = list_cells[i]
    cell_number = int(cell_number)
    delay = 1
    collapse_to = 5
    
    #### Load the sparse part
    filename_images_sparse = images_folder + 'images' + quality + stimuli_type_sparse + image_format
    filename_h1kernel_sparse = kernels_folder + 'h1' + str(cell_number) + stimuli_type_sparse + kernel_format
    filename_h2kernel_sparse = kernels_folder + 'h2' + str(cell_number) + stimuli_type_sparse + kernel_format
    
    h1_sparse = np.load(filename_h1kernel_sparse)
    h2_sparse = np.load(filename_h2kernel_sparse)
    
    #### Load the dense part
    filename_images_dense = images_folder + 'images' + quality + stimuli_type_dense + image_format
    filename_h1kernel_dense = kernels_folder + 'h1' + str(cell_number) + stimuli_type_dense + kernel_format
    filename_h2kernel_dense = kernels_folder + 'h2' + str(cell_number) + stimuli_type_dense + kernel_format
    
    h1_dense = np.load(filename_h1kernel_dense)
    h2_dense = np.load(filename_h2kernel_dense)
    
    kernel_size = h1_dense.shape[0]
    
    ############
    # Plotting
    ############
    cell_number = str(cell_number)
    print '****************'
    print 'plotting cell number ' + cell_number 
   #symmetric = 1
    symmetric = 2
    colorbar = True
    closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2
    aux1=-0.70 
    aux2=0.45
    
    # Plot dense
    
    directory = './figures/'
    formating='.pdf'
    title = 'simulation_regresion_h1' + quality + 'cell' + cell_number + '_'+ stimuli_type_dense
    save_filename = directory + title + formating 
     
   #plot_mutliplot_bilinear(closest_square_to_kernel, h1_dense, colorbar=colorbar, symmetric=symmetric)
    plot_mutliplot_bilinear(closest_square_to_kernel, h1_dense, colorbar=colorbar, symmetric=symmetric, aux1=aux1, aux2=aux2)
    figure = plt.gcf() # get current figure
    
    if remove_axis:
        #Remove axis 
        for i in xrange(closest_square_to_kernel):
            figure.get_axes()[i].get_xaxis().set_visible(False)
            figure.get_axes()[i].get_yaxis().set_visible(False)
        
    figure.set_size_inches(16, 12)
    plt.savefig(save_filename, dpi = 100)
    os.system("pdfcrop %s %s" % (save_filename, save_filename))
    
    if show_plot:
        plt.show()
    
    
    plot_mutliplot_bilinear(closest_square_to_kernel, h2_dense, colorbar=colorbar, symmetric=symmetric, aux1=aux1, aux2=aux2)
    title = 'simulation_regresion_h2' + quality  + 'cell' + cell_number + '_'+ stimuli_type_dense
    save_filename = directory + title + formating
    figure = plt.gcf() # get current figure
    
    if remove_axis:
    # Remove axis 
        for i in xrange(closest_square_to_kernel):
            figure.get_axes()[i].get_xaxis().set_visible(False)
            figure.get_axes()[i].get_yaxis().set_visible(False)
    
    figure.set_size_inches(16, 12)
    plt.savefig(save_filename, dpi = 100)
    os.system("pdfcrop %s %s" % (save_filename, save_filename))
    
    if show_plot:
        plt.show()
    
    
    # Plot sparse
    directory = './figures/'
    formating='.pdf'
    title = 'simulation_regresion_h1' + quality + 'cell' + cell_number + '_' + stimuli_type_sparse
    save_filename = directory + title + formating 
     
    plot_mutliplot_bilinear(closest_square_to_kernel, h1_sparse, colorbar=colorbar, symmetric=symmetric, aux1=aux1, aux2=aux2)
    figure = plt.gcf() # get current figure
    
    if remove_axis:
        #Remove axis 
        for i in xrange(closest_square_to_kernel):
            figure.get_axes()[i].get_xaxis().set_visible(False)
            figure.get_axes()[i].get_yaxis().set_visible(False)
    
    figure.set_size_inches(16, 12)
    plt.savefig(save_filename, dpi = 100)
    os.system("pdfcrop %s %s" % (save_filename, save_filename))
    
    if show_plot:
        plt.show()
    
    plot_mutliplot_bilinear(closest_square_to_kernel, h2_sparse, colorbar=colorbar, symmetric=symmetric, aux1=aux1, aux2=aux2)
    title = 'simulation_regresion_h2' + quality + 'cell' + cell_number + '_' + stimuli_type_sparse
    save_filename = directory + title + formating
    figure = plt.gcf() # get current figure
    
    if remove_axis:
    # Remove axis 
        for i in xrange(closest_square_to_kernel):
            figure.get_axes()[i].get_xaxis().set_visible(False)
            figure.get_axes()[i].get_yaxis().set_visible(False)
    
    
    figure.set_size_inches(16, 12)
    plt.savefig(save_filename, dpi = 100)
    os.system("pdfcrop %s %s" % (save_filename, save_filename))
    
    if show_plot:
        plt.show()
