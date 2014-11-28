#*************************************************************************#

# Estimates the volterra kernels for 1 cell from Cyril's data
# HOW MANY CELLS ???

#*************************************************************************#

#from functions import *
from plot_functions import plot_mutliplot_bilinear
from analysis_functions import *
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
from time import localtime
import scipy.io
from sklearn.linear_model import LinearRegression  # Import the learning algorithm


####################
# Load and process the data 
####################

## Load 
number_of_files = 6
directory = './cyrill/'
file_name = 'EP'
extension = '.mat'
stimuli_type = 'SparseNoise' # Comment the one that is not going to be used 
stimuli_type = 'DenseNoise'

# Save things or not
save_figures = True # Save the figures if True 
save_files = False  # Save the files if True 
remove_axis = True # Remove axis if True 

# Save the files to a list 
files = []
for i in xrange(number_of_files):
    file_to_retrieve = directory + file_name + str(i+1) + extension
    files.append(scipy.io.loadmat(file_to_retrieve))

    
# Kernel data and intialization
dh = 4.0  #milliseconds
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dh) 

# Empty lists to concatenate 
working_indexes_t = []
training_indexes_t = []
image_indexes_t = []
V_t = []
ims_t = []
temporal_image = []
Nlast = 0

for i, file_to_retrieve in enumerate(files):

    data = file_to_retrieve.values() # Import the data 

    ims = data[4] # The arrays with the images 
    ims = np.transpose(ims)
    ims = ims / 2.0  # Scale 
    ims = ims - 0.5  # Center 
    ims2 = ims**2
    Nside = ims.shape[2]
    
    K = np.zeros(5)
    print 'for file' + str(i) + ' printing 5 values of ims and max and min'
    for j in np.arange(4):
        K[j]= ims[j,5,5]
    print K
    print 'max = ' + str(np.max(ims)) + ' //   min = ' + str(np.min(ims))
         
    
#==============================================================================
#     frame_times = data[0] # The times at which the frames start 
#     diff_frame = np.diff(frame_times, axis=0) # This calculate the duration of each frame 
#     vm = data[2] # Extracts voltage 
# 
#     ## Pre-process the signal 
#     # sampling interval
#     sampling_time_interval = 0.102 # Obtained from Cyrills, experimental value 
#     factor = 10  # 
#     vm = downsample(vm, factor) # Sample down the signal by factor 
# 
#     first_image = int( frame_times[0] / (factor * sampling_time_interval))   # time of the first image
#     last_image = int( (frame_times[-1] + np.mean(diff_frame)) / (factor * sampling_time_interval))
#     last_image = int( (frame_times[-1] ) / (factor * sampling_time_interval)) # time of the last image 
# 
#     vm = vm[first_image:last_image] # Takes only the voltage that corresponds to images 
#         
#     ################
#     ## Data parameters 
#     ################
#         
#     #Scale and size values 
#     dt = sampling_time_interval * factor # Sampling time interval (ms)
#     dim = np.mean(diff_frame) # Duration of each image (ms)
#      
#     # Scale factors
#     input_to_image  = dt / dim # Transforms input to image
#     kernel_to_input = dh / dt  # Transforms kernel to input
#     image_to_input = dim / dt  # transforms imagen to input  
#           
#     Ntotal = vm.size  # Number of data to use (must be equal or less than vm.size)
#     #Take the minimum between the maximum and the choice
#     Ntotal = np.min((Ntotal, vm.size))
#     V = vm[0:Ntotal]
#         
#     #Size of the training set as a percentage of the data 
#     alpha = 1 #  training vs total 
#     Ntraining = int(alpha * Ntotal) 
#      
#     #Construct the set of indexes (training, test, working)
#     remove_start = int(kernel_size * kernel_to_input)  # Number of images in a complete kernel
#     working_indexes = np.arange(Ntotal)
#     training_indexes = np.arange(remove_start, Ntraining)
#        
#     #Calculate kernel
#     kernel_times = np.arange(kernel_size)
#     kernel_times = kernel_times.astype(int) # Make the values indexes 
#          
#     #Delay indexes 
#     delay_indexes = np.floor(kernel_times * kernel_to_input)
#     delay_indexes = delay_indexes.astype(int)
#      
#     #Image Indexes 
#     image_indexes = np.zeros(working_indexes.size)
#       
#     for k, index in enumerate(working_indexes):
#         index = index * dt + frame_times[0]  #Transform to ms
#         aux = np.where(frame_times > index)[0] # Indexes of the frame times thata 
#         aux = np.min(aux)
#         image_indexes[k] = int(aux)  - 1
#       
# #     image_indexes = np.zeros(working_indexes.size)
# #     image_indexes[working_indexes] = np.floor(working_indexes * input_to_image)
# #     image_indexes = image_indexes.astype(int) 
# #      
#     # Concatenate 
#     print i         
#     if i > 0:
#         image_indexes_t = np.concatenate((image_indexes_t, image_indexes + ims_t.shape[0] ))
#         ims_t = np.concatenate((ims_t, ims), axis=0)    
#         
#         working_indexes_t = np.concatenate((working_indexes_t, working_indexes + Nlast))
#         training_indexes_t = np.concatenate((training_indexes_t, training_indexes + Nlast ))
#         
#         V_t = np.concatenate((V_t, vm))
#           
#      
#     else:
#         ims_t = ims
#         working_indexes_t = np.concatenate((working_indexes_t, working_indexes))
#         training_indexes_t = np.concatenate((training_indexes_t, training_indexes))
#         image_indexes_t = np.concatenate((image_indexes_t, image_indexes))
#         V_t = np.concatenate((V_t, vm))
#     
#     Nlast += Ntotal
#     
#     # show files increase
#     print 'last image index', image_indexes_t[-1] 
#     print 'image size ', ims_t.shape[0]
#     print 'V', V_t.shape, 
#     print 'image_indexes', image_indexes_t.shape
#     print 'training_indexes', training_indexes_t.shape
#     print 'working_indexes', working_indexes_t.shape
#     print '------------------------------------------'
# 
# 
# 
# # Erase unused references  
# ims = None
# vm = None
# ims2 = None 
# V = None
# training_indexes = None
# working_indexes = None
# image_indexes = None
# files = None
# 
# # Make things indexes
# training_indexes = training_indexes_t.astype(int)
# image_indexes = image_indexes_t.astype(int)
# working_indexes = working_indexes_t.astype(int)
# 
# 
# # take out the spikes 
# #treshold = -60 
# #V_t [ V_t > treshold] = treshold 
# 
# mean_t = np.mean(V_t)
# V_t = V_t - mean_t
# 
# V = V_t 
# ims = ims_t
# ims2 = ims **  2 
# 
# ims_t = None
# V_t= None
# 
# ########################
# # Calculate Regression 
# ########################
# 
# # Number of parameters
# Nparameters = Nside*Nside*2
# 
# # Create a vector with the indexes of the elements after the image 
# extract = np.arange(0, training_indexes.size, int(image_to_input), dtype=int)
# training_indexes = training_indexes[extract]
# 
# # Initialize the kernels 
# h1 = np.zeros((kernel_size, Nside, Nside))
# h2 = np.zeros((kernel_size, Nside, Nside))
# 
# # Targets
# Y = V[training_indexes]
# # Create the matrix of training
# X = np.zeros((training_indexes.size, Nparameters))
# 
# print 'X shape', X.shape
# print 'Y shape', Y.shape
# 
# 
# for tau, delay_index in enumerate(delay_indexes):
#     print 'tau', tau
#     print 'Creating the matrix X'
#     for i, index in enumerate(training_indexes):
# 
#         delay = image_indexes[index - delay_index]
#         f1 = np.reshape(ims[delay, ...], Nside*Nside)
#         f2 = np.reshape(ims2[delay, ...], Nside*Nside)
#         X[i, :] = np.concatenate((f1,f2))
# 
# 
#     predictor = LinearRegression(copy_X=False, fit_intercept=False)
#     predictor.fit(X, Y)
#  
# 
#     ## Order parameters
#     parameters = predictor.coef_
#     h1_dis = parameters[0:Nparameters / 2]
#     h2_dis = parameters[Nparameters / 2 :]
# 
#     h1[tau,...] = h1_dis.reshape(Nside,Nside)
#     h2[tau,...] = h2_dis.reshape(Nside,Nside)
# 
# 
# ############
# # Plotting 
# ############
# 
# if save_figures:
# 
#     symmetric = 0
#     colorbar = True 
#     closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2
#     
#     directory = './figures/'
#     formating='.pdf'
#     title = 'real_regresion_h1' + stimuli_type
#     save_filename = directory + title + formating 
#      
#     plot_mutliplot_bilinear(closest_square_to_kernel, h1, colorbar=colorbar, symmetric=symmetric)
#     figure = plt.gcf() # get current figure
#     
#     if remove_axis:
#         # Remove axis 
#         for i in xrange(closest_square_to_kernel):
#             figure.get_axes()[i].get_xaxis().set_visible(False)
#             figure.get_axes()[i].get_yaxis().set_visible(False)
#     
#     figure.set_size_inches(16, 12)
#     plt.savefig(save_filename, dpi = 100)
#     os.system("pdfcrop %s %s" % (save_filename, save_filename))
#     
#     
#     plt.show()
#     
#     
#     plot_mutliplot_bilinear(closest_square_to_kernel, h2, colorbar=colorbar, symmetric=symmetric)
#     title = 'real_regresion_h2' + stimuli_type
#     save_filename = directory + title + formating
#     figure = plt.gcf() # get current figure
#     
#     if remove_axis:
#         # Remove axis 
#         for i in xrange(closest_square_to_kernel):
#             figure.get_axes()[i].get_xaxis().set_visible(False)
#             figure.get_axes()[i].get_yaxis().set_visible(False)
#     
#     figure.set_size_inches(16, 12)
#     plt.savefig(save_filename, dpi = 100)
#     os.system("pdfcrop %s %s" % (save_filename, save_filename))
#     
#     
#     plt.show()
#==============================================================================
#==============================================================================
#     
# ############
# # Saving 
# ############
# if save_figures:
# 
#     directory = './kernels/'
#     formating='.npy'
#     title = 'real_regresion_h1'
#     save_filename1 = directory + stimuli_type + title + formating
#     
#     np.save(save_filename1, h1)
#     
#     directory = './kernels/'
#     formating='.npy'
#     title = 'real_regresion_h2'
#     save_filename2 = directory + stimuli_type + title + formating
#     
#
#     np.save(save_filename2, h2)
#==============================================================================