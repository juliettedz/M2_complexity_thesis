'''
Created on Jun 4, 2014

@author: ramon
'''

from functions import *
from old_functions import sta_v_old_2
from plot_functions import plot_mutliplot_bilinear
from analysis_functions import *
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from time import localtime


####################
# Load the data 
####################
folder = './data/'
filename = 'exc_2000'
filename = '3000_21'
filename_vm = folder +  'vm_' + filename + '.pickle'
filename_images = folder + 'images_'+ filename + '.pickle'

f = open(filename_vm,'rb')
vm = cPickle.load(f)
f = open(filename_images,'rb')
ims = cPickle.load(f) 

# Scale and normalize 
ims = ims / 100 #Scale 
ims = ims - 0.5 # Center 

ims2 = ims**2
Nside = ims.shape[2]

print 'voltate shape', np.shape(vm)
print 'image shape', np.shape(ims)

####
#
#############

#Scale and size values 
dt = 1.0  #milliseconds
dim = 21.0 # milliseconds
dh = 7.0 #milliseconds
kernel_size = 20

# Scale factors 
input_to_image  = dt / dim # Transforms input to image  
kernel_to_input = dh / dt  # Transforms kernel to input 

## Input preprocesing 
vm = downsample(vm,dt)

# Take the data that is going to be use from the total data
Ntotal = 2 * 10**(5) # Number of data to use
Ntotal = vm.size 
# Take the minimum between the maximum and the choice
Ntotal = np.min((Ntotal, vm.size))
V = vm[0:Ntotal] 
V = V - np.mean(V)
#std = np.std(V)
#V = V / std 


###
# Time traces 
###
image_to_input = dim/ dt  # Transforms image_to_input 
# Index to analyze 
x = 5
y = 5
time_window = 200


positive_time_trace = positive_time_trace(x,y, time_window, image_to_input, V, ims)
negative_time_trace = negative_time_trace(x,y, time_window, image_to_input, V, ims)

plt.plot(positive_time_trace, label='positive')
plt.hold('on')
plt.plot(negative_time_trace, label='negative')
plt.legend()
plt.show()



