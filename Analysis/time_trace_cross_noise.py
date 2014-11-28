'''
Created on Nov 21, 2014

@author: juliette
'''

import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os 
from error_functions import calculate_prediction

## Files 
images_folder = './data/'
kernels_folder = './kernels/'
quality = '_15000_21_'
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'

remove_axis = False
cell_number = 2
show_plot = True
time_window = 600 #ms

directory = './figures/'
formating='.pdf'
title = 'crossNoises_small' + quality + 'cell' + str(cell_number) 
save_filename = directory + title + formating 

###############
# preparatioin of calculations
###############

#Scale and size values
dt = 1.0  #milliseconds
dim = 21.0 # milliseconds
dh = 7.0 #milliseconds
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dh)

# Scale factors
input_to_image  = dt / dim # Transforms input to image
kernel_to_input = dh / dt  # Transforms kernel to input
image_to_input = dim / dt

Number_of_images = 10000 # Number of images 
Ntraining = int(Number_of_images * dim)
working_indexes = np.arange(Ntraining)

remove_start = int(kernel_size * kernel_to_input)  # Number of images in a complete kernel
training_indexes = np.arange(remove_start, Ntraining)

# Calculate kernel
kernel_times = np.arange(kernel_size)
kernel_times = kernel_times.astype(int) # Make the values indexes

# Delay indexes
delay_indexes = np.floor(kernel_times * kernel_to_input)
delay_indexes = delay_indexes.astype(int)

# Image Indexes
image_indexes = np.zeros(Ntraining)
image_indexes[working_indexes] = np.floor(working_indexes * input_to_image)
image_indexes = image_indexes.astype(int)

## Load images
# Sparse
filename_images_sparse = images_folder + 'images' + quality + stimuli_type_sparse + image_format
f = open(filename_images_sparse,'rb' )
ims_sparse = cPickle.load(f)
ims_sparse = ims_sparse / 100
ims_sparse = ims_sparse - 0.5

f.close()

# Dense
filename_images_dense = images_folder + 'images' + quality + stimuli_type_dense + image_format
f = open(filename_images_dense,'rb' )
ims_dense = cPickle.load(f)
ims_dense = ims_dense / 100
ims_dense = ims_dense - 0.5

f.close()

h0 = 0 

print 'CALCULATE new time traces'

#### Load the sparse part
filename_h1kernel_sparse = kernels_folder + 'h1' + str(cell_number) + stimuli_type_sparse + kernel_format
filename_h2kernel_sparse = kernels_folder + 'h2' + str(cell_number) + stimuli_type_sparse + kernel_format

h1_sparse = np.load(filename_h1kernel_sparse)
h2_sparse = np.load(filename_h2kernel_sparse)


#### Load the dense part 

filename_h1kernel_dense = kernels_folder + 'h1' + str(cell_number) + stimuli_type_dense + kernel_format
filename_h2kernel_dense = kernels_folder + 'h2' + str(cell_number) + stimuli_type_dense + kernel_format

h1_dense = np.load(filename_h1kernel_dense)
h2_dense = np.load(filename_h2kernel_dense)

# Calculate the convolutions
dense_over_sparsefield_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                     kernel_to_input, h0, h1_sparse, h2_sparse, ims_dense, ims_dense**2)


sparse_over_densefield_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                     kernel_to_input, h0, h1_dense, h2_dense, ims_sparse, ims_sparse**2)


# cut a window of time_window
new_time_trace_SD = dense_over_sparsefield_convoluted[1: time_window]
new_time_trace_DS = sparse_over_densefield_convoluted[1: time_window]

print 'PLOT new time traces'

figure = plt.figure
plt.subplot(211)    
plt.plot(new_time_trace_SD)
plt.ylabel('SN^DN')
plt.xlabel('ms')
plt.title('RF ^ St')
plt.subplot(212) 
plt.plot(new_time_trace_DS)
plt.ylabel('DN^SN')
plt.xlabel('ms')


figure = plt.gcf() # get current figure

if remove_axis:
    #Remove axis 
    for i in [1,2]:  #10 = number of plots in the figure
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

if show_plot:
    plt.show()

