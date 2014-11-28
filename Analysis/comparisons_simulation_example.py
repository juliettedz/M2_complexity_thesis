# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/juliette/.spyder2/.temp.py
"""

import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from analysis_functions import *
from time import localtime


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

#choose what to plot : to choose the time: 1=early, 2=right, 3=late
cell_number = 1
time_choice = 2

#choose parameters of the plot:
symmetric = 3
remove_axis = False
show_plot = True
aux1 = -0.45
aux2 = 0.7

 ###########################################################
#                      RECEPTIVE FIELD                      #
############################################################

 ############
# Load Data
############

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

#### choice of time of observation
# time = kernel_size / 2
if time_choice == 1:
    time = kernel_size /4
elif time_choice == 2:
    time = kernel_size / 2
elif time_choice == 2:
    time = kernel_size *3/ 4
time=int(time)

################ 
# create plots, used to be in plot_functions
###############

cell_number = str(cell_number)
print '****************'
print 'plotting comparisons for: cell number ' + cell_number 

directory = './figures/'
formating='.pdf'
title = 'comparison' + quality + 'cell' + cell_number + '_time_'+ str(time)
save_filename = directory + title + formating 
  
################ 
# create plots, used to be in plot_functions
###############

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

#plot RF
figure = plt.figure

print '*** plotting Receptive Field' 
##### For h1_sparse #####
ims = h1_sparse
im = plt.subplot(521)
im = plt.imshow(ims[time,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
im = plt.ylabel('h1')
im = plt.title('SN')
plt.colorbar()


##### For h2_sparse #####
ims = h2_sparse
im = plt.subplot(523)
im = plt.imshow(ims[time,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
im = plt.ylabel('h2')
plt.colorbar()

##### For h1_dense #####
ims = h1_dense
im = plt.subplot(527)
im = plt.imshow(ims[time,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
im = plt.ylabel('h1')
im = plt.title('DN')
plt.colorbar()

##### For h2_dense #####
ims = h2_dense
im = plt.subplot(529)
im = plt.imshow(ims[time,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
im = plt.ylabel('h2')
plt.colorbar() 

figure = plt.gcf()

if remove_axis:
    #Remove axis 
    for i in xrange(4):  #10 = number of plots in the figure
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

 ######################################################
#                     TIME TRACES                      #
#######################################################

folder = './data/'
cell = '_cell_' + str(cell_number) 
quality = '_15000_21_'
stimuli_type = 'SparseNoise'
file_format = '.pickle'

filename_vm = folder + 'vm' + cell + quality + stimuli_type + file_format
filename_images = folder + 'images' + quality + stimuli_type + file_format

# Save things or not
save_figures = True # Save the figures if True 
remove_axis = False # Remove axis if True 

f = open(filename_vm,'rb')
vm = cPickle.load(f)
f = open(filename_images,'rb')
ims = cPickle.load(f) 

# Scale and normalize 
ims = ims / 100 #Scale 
ims = ims - 0.5 # Center 

ims2 = ims**2
Nside = ims.shape[2]

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


###############################
# Plotting 
###############################

print '*** plotting time-trace for SPARSE NOISE' 

# Let's first do a 10 x 10 grid with the positive traces 
time_window = 200  # In ms     
x = 3
y = 3
size = 4

positive = positive_time_trace(x,y, time_window, image_to_input, V, ims)
negative = negative_time_trace(x,y, time_window, image_to_input, V, ims)

plt.subplot(522)
plt.plot(positive)
plt.hold(True)
plt.plot(negative)
plt.xlabel('ms')


figure = plt.gcf() # get current figure


###########      DENSE NOISE        ######################
stimuli_type = 'DenseNoise'
file_format = '.pickle'

filename_vm = folder + 'vm' + cell + quality + stimuli_type + file_format
filename_images = folder + 'images' + quality + stimuli_type + file_format

# Save things or not
save_figures = True # Save the figures if True 
remove_axis = False # Remove axis if True 

f = open(filename_vm,'rb')
vm = cPickle.load(f)
f = open(filename_images,'rb')
ims = cPickle.load(f) 

# Scale and normalize 
ims = ims / 100 #Scale 
ims = ims - 0.5 # Center 

ims2 = ims**2
Nside = ims.shape[2]

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


###############################
# Plotting 
###############################

print '*** plotting time-trace for DENSE NOISE' 

# Let's first do a 10 x 10 grid with the positive traces 
time_window = 200  # In ms     
x = 3
y = 3
size = 4

positive = positive_time_trace(x,y, time_window, image_to_input, V, ims)
negative = negative_time_trace(x,y, time_window, image_to_input, V, ims)

plt.subplot(528)
plt.plot(positive)
plt.hold(True)
plt.plot(negative)
plt.xlabel('ms')



###################### SAVE AND REMOVE AXIS

figure = plt.gcf() # get current figure

if remove_axis:
    #Remove axis 
    for i in [5,6]:  #10 = number of plots in the figure
        figure.get_axes()[i].get_xaxis().set_visible(False)
        figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

if show_plot:
    plt.show()



