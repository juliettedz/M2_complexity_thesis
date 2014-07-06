'''
Created on Jun 2, 2014

@author: ramon
'''

from plot_functions import plot_mutliplot_bilinear
from analysis_functions import downsample, sta_v
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from time import localtime

####################
# Load the data
####################
folder = './data/'
cell_number = 8
cell = '_cell_' + str(cell_number) 
quality = '_3000_21_'
stimuli_type = 'SparseNoise'
stimuli_type = 'DenseNoise'
file_format = '.pickle'
filename_vm = folder + 'vm' + cell + quality + stimuli_type + file_format
filename_images = folder + 'images'+ quality + stimuli_type + file_format

# Save things or not
remove_axis = True

f = open(filename_vm,'rb')
vm = cPickle.load(f)
f = open(filename_images,'rb' )
ims = cPickle.load(f)
ims = ims / 100
ims = ims - 0.5
#ims = ims - 50.0
ims2 = ims**2
Nside = ims.shape[2]

f.close()

##########################33
# Parameters of the data 
##########################33

#Scale and size values
dt = 1.0  # time sampling (ms)
dim = 21.0 # duration of the image (ms)
dh = 7.0 # resolution of the kernel (ms)

kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dh) 

# Scale factors
input_to_image  = dt / dim # Transforms input to image
kernel_to_input = dh / dt  # Transforms kernel to input
image_to_input = dim / dt  # transforms imagen to input 

## Input preprocesing
vm = downsample(vm,dt)  

# Take the percentage of the total that is going to be used 
percentage = 0.30
Ntotal = int(percentage * vm.size)

# Take the minimum between the maximum and the choice
Ntotal = np.min((Ntotal, vm.size))
V = vm[0:int(Ntotal)]
vm = None # Liberate memory

# Size of the training set as a percentage of the data
alpha = 0.95 #  training vs total
Ntraining = int(alpha * Ntotal)

# Construct the set of indexes (training, test, working)
Ntest = 10000
remove_start = int(kernel_size * kernel_to_input)  # Number of images in a complete kernel
Ntest = np.min((Ntest, Ntotal - Ntraining)) # Take Ntest more examples to test, or the rest available
working_indexes = np.arange(Ntotal)
working_indexes = working_indexes.astype(int)

training_indexes = np.arange(remove_start, Ntraining)
test_indexes = np.arange(Ntraining,Ntraining + Ntest)
test_indxes = test_indexes.astype(int)

# Calculate kernel
kernel_times = np.arange(kernel_size)
kernel_times = kernel_times.astype(int) # Make the values indexes

# Delay indexes
delay_indexes = np.floor(kernel_times * kernel_to_input)
delay_indexes = delay_indexes.astype(int)

# Image Indexes
image_indexes = np.zeros(working_indexes.size)
image_indexes[working_indexes] = np.floor(working_indexes * input_to_image)
image_indexes = image_indexes.astype(int)

# Center and normalize the output 
mean = np.mean(V[training_indexes])
V = V - mean
#std = np.std(V)
#V = V / std
#V = V / (np.max(V) - np.min(V))
#V  = V / (np.max(np.abs(V)))

############
# Calculate STA
############
print 'file name:',  quality+ stimuli_type
print 'Examples used to calculate it :', training_indexes.size
print 'kernel size', kernel_size
verbose = True # Whether we want the delays to be display or not

sta = sta_v(V, ims, training_indexes, delay_indexes, image_indexes, kernel_to_input, input_to_image, kernel_times, verbose=verbose)

############
# Plotting and saving 
############
symmetric = 1
colorbar = True
closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2
plot_mutliplot_bilinear(closest_square_to_kernel, sta)

directory = './figures/'
formating='.pdf'
title = 'STA' + quality + stimuli_type
save_filename = directory + title + formating 

figure = plt.gcf() # get current figure

if remove_axis:
    #Remove axis 
    for i in xrange(closest_square_to_kernel):
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()

