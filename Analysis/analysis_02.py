################################################################################
## The kernels files for cells are read in this files and the SI  
## and the gains are calculated and plotted 
################################################################################

import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os 
from error_functions import calculate_prediction

## Files 
images_folder = './data/'
kernels_folder = './kernels/'
quality = '_3000_21_'
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'


number_of_cells = 21
SI_sparse = np.zeros(number_of_cells)
SI_dense = np.zeros(number_of_cells)


#Scale and size values
dt = 1.0  #milliseconds
dim = 21.0 # milliseconds
dh = 5.0 #milliseconds
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dh)

# Scale factors
input_to_image  = dt / dim # Transforms input to image
kernel_to_input = dh / dt  # Transforms kernel to input
image_to_input = dim / dt

Number_of_images = 2000 # Number of images 
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


for cell_number in xrange(number_of_cells):
    cell = '_cell_' + str(cell_number) 
    print 'cell', cell
    
    
    
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
    aux_zeros = np.zeros(np.shape(h1_sparse))
    
    h1_sparse_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                         kernel_to_input, h0, h1_sparse, aux_zeros, ims_sparse, ims_sparse**2)
    
    h2_sparse_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                         kernel_to_input, h0, aux_zeros, h2_sparse, ims_sparse, ims_sparse**2)
    
    h1_dense_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                         kernel_to_input, h0, h1_dense, aux_zeros, ims_dense, ims_dense**2)
    
    h2_dense_convoluted = calculate_prediction(training_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                         kernel_to_input, h0, aux_zeros, h2_dense, ims_dense, ims_dense**2)
    
    ###  Calculate the SI's 
    SI_sparse[cell_number] = np.sum(h1_sparse_convoluted*2) / np.sum( h2_sparse_convoluted**2 + h1_sparse_convoluted**2)
    SI_dense[cell_number] = np.sum(h1_dense_convoluted**2 ) / np.sum( h2_dense_convoluted**2 + h1_dense_convoluted**2)
    
    

### ### ### 
# Plot and save 
### ### ### 
#plt.subplot(1,2,1)
max = np.max((np.max(SI_sparse), np.max(SI_dense)))
plt.plot(SI_sparse , SI_dense , '*')
plt.xlabel('SI* SN')
plt.ylabel('SI* DN')
plt.xlim([0,1])
plt.ylim([0,1])

t = np.linspace(0,1,100)
plt.plot(t,t, 'k')

# Change the font to font size
fontsize = 25
figure = plt.gcf()
ax = figure.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)


directory = './figures/'
formating='.pdf'
title = 'SI_convoluted'
save_filename = directory + title + formating 

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()
