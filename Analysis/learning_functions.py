import numpy as np
import matplotlib.pyplot as plt 
from scipy import nanmean
from time import localtime
from math import floor

###########################
# Learning functions 
###########################

def hypothesis(index, kernel_times, delay_indexes, image_indexes, input_to_image, kernel_to_input, h0, h1, h2, ims, ims2):
    '''
    Calculates the convolution between the kernel and the image. 
    Parameters:

    '''      
    delay = image_indexes[index - delay_indexes] 
   
    # Do the calculation    
    result = np.sum( h1[kernel_times,...] * ims[delay,...] + h2[kernel_times,...] * ims2[delay,...])
    
    return result + h0


### Learning Functions 

## Normal SGD 
############################################################################################################

def update_kernel(learning_rate, error, index, delay_indexes, image_indexes, input_to_image, kernel_to_input, kernel_times, h0, h1, h2, ims, ims2): 
    '''
    update kernel 
    
    '''
    delay = image_indexes[index - delay_indexes] 
       
    # Update the values 
    h1[kernel_times,:,:] += learning_rate * error * ims[delay,:,:]
    h2[kernel_times,:,:] += learning_rate * error * ims2[delay,:,:]
    
        
    return h1, h2

def learn_kernel_iterations(iterations, training_indexes, delay_indexes, image_indexes, V, learning_rate, kernel_times, 
                           input_to_image, kernel_to_input, h0, h1, h2, ims, ims2, verbose=False, modulus=10):
    '''
    Method that learns the volterra kernel of that predicts a given signal V
    it returns the kernel parameters h0, h1, h2 
    '''
  
    
    for i in range(iterations):    
        permutation = np.random.choice(training_indexes, training_indexes.size,replace=False) # Shuffle the data 
        for index in permutation: # Loop over training examples 
            error = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                  kernel_to_input, h0, h1, h2, ims, ims2)
            
            # Update the kernel with that error 
            h1, h2 = update_kernel(learning_rate, error, index, delay_indexes, image_indexes, input_to_image, 
                                   kernel_to_input, kernel_times, h0, h1, h2, ims, ims2)
            h0 = h0 + learning_rate * error
    
    return h0, h1, h2 


def learn_kernel_tolerance(tolerance, training_indexes, delay_indexes, image_indexes, V, learning_rate, kernel_times, 
                           input_to_image, kernel_to_input, h0, h1, h2, ims, ims2, verbose=False, modulus=10):
    '''
    Documentation here 
    '''
    print 'Tolerance Normal Algorithm'
    
    total_cum_error = training_indexes.size
    iteration = 0

    
    while((total_cum_error / training_indexes.size) > tolerance): #General loop
        
        iteration += 1  # Count the number of iterations 
        permutation = np.random.choice(training_indexes, training_indexes.size, replace=False) # Shuffle the data 
        
        for index in permutation: # Loop over training examples 
        #for index in training_indexes:
            # calculate the error
            epsilon = 0.0005
            error = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                  kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            h1, h2 = update_kernel(learning_rate, error, index, delay_indexes, image_indexes, input_to_image, 
                                   kernel_to_input, kernel_times, h0, h1, h2, ims, ims2)
            
            h0 = h0 + learning_rate * error
         
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, kernel_times, delay_indexes, image_indexes,
                                                 input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            h1[ np.abs(h1) < epsilon] = 0.0
            h2[ np.abs(h2) < epsilon] = 0.0
            
        if (verbose == True):
            # Print information regarding the progress of the algorithm
            print '*********************************'
            print 'iteration', iteration 
            print 'h0=', h0
            print 'total cumulative error', total_cum_error
            print 'total cumulative error normalized', total_cum_error / training_indexes.size
            print 'total cumulative error change', total_cum_error - AUX
            print 'cumulative ratio', total_cum_error / AUX
            print 'learning_rate', learning_rate 
            print '*********************************'
    
    return h0, h1, h2 
