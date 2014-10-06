from analysis_functions import downsample
from plot_functions import *
from store_functions import store_X, store_kernel_numpy
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os 


from sklearn.linear_model import LinearRegression  # Import the learning algorithm


####################
# Load the data
####################
folder = './data/'  #** Ju **# ajouter FFI_testDataBig_____
cell_number = 8         #** Ju **# model: x cells, then record from less cells and then analyse even less (= cell_number) cells
cell = '_cell_' + str(cell_number) 
quality = '_3000_21_'
stimuli_type = 'SparseNoise' # Comment the one that is not going to be used 
stimuli_type = 'DenseNoise'
file_format = '.pickle'
filename_vm = folder + 'vm' + cell + quality + stimuli_type + file_format
filename_images = folder + 'images'+ quality + stimuli_type + file_format

# Save things or not
save_figures = True # Save the figures if True 
save_files = False  # Save the files if True 
remove_axis = False # Remove axis if True 

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
dim = 21.0 # duration of the image (ms)  #** Ju **# comes from the experiment
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

########################
# Calculate Regression 
########################

# Number of parameters
Nparameters = Nside*Nside*2

# Create a vector with the indexes of the elements after the image 
extract = np.arange(0, training_indexes.size, int(image_to_input), dtype=int)
training_indexes = training_indexes[extract]

# Initialize the kernels 
h1 = np.zeros((kernel_size, Nside, Nside))
h2 = np.zeros((kernel_size, Nside, Nside))

# Targets
Y = V[training_indexes]
# Create the training matrix 
X = np.zeros((training_indexes.size, Nparameters))
 
print 'X shape', X.shape
print 'Y shape', Y.shape
print 'file = ', filename_vm
 
for tau, delay_index in enumerate(delay_indexes):
    print 'tau', tau
    print 'Creating the matrix X'
    
    for i, index in enumerate(training_indexes):
 
        delay = image_indexes[index - delay_index]
        f1 = np.reshape(ims[delay, ...], Nside*Nside)
        f2 = np.reshape(ims2[delay, ...], Nside*Nside)
        X[i, :] = np.concatenate((f1,f2))
 
    # Store matrix X if covariances want to be calculated 
    #store_X(X, tau, filename)
    
    # Making the predictions and firring 
    predictor = LinearRegression(copy_X=False, fit_intercept=False) 
    predictor.fit(X, Y)
    
    # Extract the parameters 
    parameters = predictor.coef_
    # Order them in simple and complex 
    h1_dis = parameters[0:Nparameters / 2]
    h2_dis = parameters[Nparameters / 2 :]
    # Store them in proper shape 
    h1[tau,...] = h1_dis.reshape(Nside,Nside)
    h2[tau,...] = h2_dis.reshape(Nside,Nside)
 
 

############
# Plotting
############
if save_figures:
    symmetric = 0
    colorbar = True
    closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2
    
    directory = './figures/'
    formating='.pdf'
    title = 'data_regresion_h1' + quality + stimuli_type
    save_filename = directory + title + formating 
     
    plot_mutliplot_bilinear(closest_square_to_kernel, h1, colorbar=colorbar, symmetric=symmetric)
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
    
    
    plot_mutliplot_bilinear(closest_square_to_kernel, h2, colorbar=colorbar, symmetric=symmetric)
    title = 'data_regresion_h2' + quality+ stimuli_type
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
    
    plt.show()
    
###############
# Saving 
###############
if save_files:
    store_kernel_numpy(kernel_size, h1, h2, cell_number, stimuli_type)
