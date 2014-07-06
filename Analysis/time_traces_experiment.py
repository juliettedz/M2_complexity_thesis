####################
# Import functions 
####################

from analysis_functions import downsample, positive_time_trace, negative_time_trace
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os
import scipy.io

####################
# Load and process the data 
####################

## Load 
number_of_files = 6
directory = './cyrill/'
file_name = 'EP'
extension = '.mat'

# Save the files to a list 
files = []
for i in xrange(number_of_files):
    file_to_retrieve = directory + file_name + str(i+1) + extension
    files.append(scipy.io.loadmat(file_to_retrieve))

    
# Kernel parameters 
dh = 7.0  # ms 
kernel_size = 20   #  Size of the kernel in dh units

time_window = 200  # Duration of the time window (ms)

# Grid parameters 
left_corner_x = 0
left_corner_y = 0 
size = 10  # 10 covers the whole grid
    
count = 0    

for x in xrange(left_corner_x, left_corner_x + size):
    for y in xrange(left_corner_y, left_corner_y + size):
        count += 1
    
        # Here we store the traces
        positive_trace = np.zeros((number_of_files,time_window)) # Create the vector where the trace will be stored 
        negative_trace = np.zeros((number_of_files,time_window)) # Create the vector where the trace will be stored 
        
        for i, file_to_retrieve in enumerate(files):
        
            data = file_to_retrieve.values() # Import the data 

            ims = data[4] # The arrays with the images 
            ims = np.transpose(ims)
            ims = ims / 2.0  # Scale 
            ims = ims - 0.5  # Center 
            ims2 = ims**2
            Nside = ims.shape[2]
            
            frame_times = data[0] # The times at which the frames start 
            diff_frame = np.diff(frame_times, axis=0) # This calculate the duration of each frame 
            vm = data[2] # Extracts voltage 
        
            ## Pre-process the signal 
            # sampling interval
            sampling_time_interval = 0.102 # Obtained from Cyrills, experimental value 
            factor = 10  # 
            vm = downsample(vm, factor) # Sample down the signal by factor 
        
            first_image = int( frame_times[0] / (factor * sampling_time_interval))   # time of the first image
            last_image = int( (frame_times[-1] + np.mean(diff_frame)) / (factor * sampling_time_interval))
            last_image = int( (frame_times[-1] ) / (factor * sampling_time_interval)) # time of the last image 
        
            vm = vm[first_image:last_image] # Takes only the voltage that corresponds to images 
                
            ################
            ## Data parameters 
            ################
                
            #Scale and size values 
            dt = sampling_time_interval * factor # Sampling time interval (ms)
            dim = np.mean(diff_frame) # Duration of each image (ms)
             
            #Scale factors 
            input_to_image  = dt / dim # Transforms input to image  
            kernel_to_input = dh / dt  # Transforms kernel to input 
                  
            Ntotal = vm.size  # Number of data to use (must be equal or less than vm.size)
            #Take the minimum between the maximum and the choice
            Ntotal = np.min((Ntotal, vm.size))
            V = vm[0:Ntotal]
            
            ############
            # Calculate traces for each of the file pieces 
            ############
            print 'file name:', file_name
            print 'kernel size', kernel_size
            print '-----------------'
            print 'index = ', i
            
            image_to_input = dim/ dt  # Transforms image_to_input 
            
            # Calculate the trace for that piece of data
            positive_trace[i,:] =  positive_time_trace(x, y , time_window, image_to_input, V, ims)
            negative_trace[i,:] =  negative_time_trace(x, y , time_window, image_to_input, V, ims)
            
        positive_trace = np.mean(positive_trace, axis=0)
        negative_trace = np.mean(negative_trace, axis=0)
        
        
        # Plot that particular cell 
        print x, y
        plt.subplot(size, size, count)
        plt.plot(positive_trace, label='positive')
        #plt.title('V vs time')
        #plt.grid()
        plt.hold(True)
        plt.plot(negative_trace, label='negative')
        #plt.legend()
        plt.hold(False)

directory = './figures/'
formating='.pdf'
title = 'EPtraces' 
save_filename = directory + title + formating 


figure = plt.gcf() # get current figure

# Remove axis 
for i in xrange(size ** 2):
    figure.get_axes()[i].get_xaxis().set_visible(False)
    figure.get_axes()[i].get_yaxis().set_visible(False)

figure.set_size_inches(16, 12)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()


