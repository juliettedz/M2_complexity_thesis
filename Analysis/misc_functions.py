'''
Created on Apr 29, 2014

@author: ramon
'''

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

## Momentum 
############################################################################################################

def update_kernel_momentum(momentum, learning_rate, error, index, delay_indexes, image_indexes, 
                           kernel_times, h0, h1, h2, ims, ims2, Dh1, Dh2): 
    '''
    update kernel 
    
    '''
    
    delay = image_indexes[index - delay_indexes] 
    
    # Calculate the change 
    Dh2[kernel_times,:,:] = learning_rate * error * ims2[delay,:,:] + momentum * Dh2[kernel_times,:,:]
    Dh1[kernel_times,:,:] = learning_rate * error * ims[delay,:,:] + momentum * Dh1[kernel_times,:,:]
       
    # Update the values 
    h2[kernel_times,:,:] += Dh2[kernel_times,:,:] 
    h1[kernel_times,:,:] += Dh1[kernel_times,:,:]
            
    return h1, h2, Dh1, Dh2 


def learn_kernel_tolerance_momentum(tolerance, momentum, training_indexes, delay_indexes, image_indexes,  V, learning_rate, 
                                    kernel_times, input_to_image, kernel_to_input, h0, h1, h2, ims, ims2, verbose=False, modulus=10):
    '''
    Documentation here 
    '''
    
    print 'Momentum algorithms'

    # Initialize momentum
    Dh1 = np.zeros(np.shape(h1))
    Dh2 = Dh1 
    
    total_cum_error = training_indexes.size
    iteration = 0
    
    while((total_cum_error / training_indexes.size) > tolerance): #General loop
        
        iteration += 1  # Count the number of iterations 
        
        # Learning rate 
        #exponent = (np.log10(iteration) / 0.4) + 3.9
        #learning_rate = 10 ** (-exponent)
         
        permutation = np.random.choice(training_indexes, training_indexes.size, replace=False) # Shuffle the data 
        
        for index in permutation: # Loop over training examples 
            # calculate the error 
            error = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                  kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            h1, h2, Dh1, Dh2 = update_kernel_momentum(momentum, learning_rate, error, index, delay_indexes, image_indexes, 
                                                      kernel_times, h0, h1, h2, ims, ims2, Dh1, Dh2)
            #h0 = h0 + learning_rate * error
         
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, kernel_times, delay_indexes, image_indexes,
                                                 input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            
            if (verbose == True):
                # Print information regarding the progress of the algorithm
                print '*********************************'
                print 'iteration', iteration 
                print 'total cumulative error', total_cum_error
                print 'total cumulative error normalized', total_cum_error / training_indexes.size
                print 'total cumulative error change', total_cum_error - AUX
                print 'cumulative ratio', total_cum_error / AUX 
                print 'learning_rate', learning_rate
                print 'momentum', momentum
                print '*********************************'
    
    return h0, h1, h2 

## Ada-delta 
############################################################################################################


def update_kernel_adadelta(epsilon, decay_rate, error, index, image_indexes, delay_indexes,
                            kernel_times, h0, h1, h2, ims, ims2,Dg1, Dg2, Dh1, Dh2, constant): 
    '''
    update kernel 
    
    '''
    delay = image_indexes[index - delay_indexes] 
    # Make them indexes 
    delay = delay.astype(int)     
    
    ## Calculation
    g1 = error * ims[delay,:,:]
    #print 'g1,', g1
    g2 = error * ims2[delay,:,:] # 1 ) Gradients
       
    Dg1[kernel_times,:,:] = decay_rate * Dg1 + (1 - decay_rate) * (g1**2) 
    Dg2[kernel_times,:,:] = decay_rate * Dg2 + (1 - decay_rate) * (g2**2) # 2) Accumulate gradients 
    
    RMS_update1 = np.sqrt(Dh1[kernel_times,:,:] + epsilon)
    RMS_update2 = np.sqrt(Dh2[kernel_times,:,:] + epsilon)
    RMS_g1 = np.sqrt(Dg1[kernel_times,:,:] + epsilon)
    RMS_g2 = np.sqrt(Dg2[kernel_times,:,:] + epsilon) # 3) Calculate RMS values
      
    update1 = (RMS_update1 / RMS_g1) * g1
    update2 = (RMS_update2 / RMS_g2) * g2 # 4) Calculate updates
       
    Dh1[kernel_times,:,:] = decay_rate * Dh1[kernel_times,:,:] + (1 - decay_rate) * (update1**2) 
    Dh2[kernel_times,:,:] = decay_rate * Dh2[kernel_times,:,:] + (1 - decay_rate) * (update2**2) # 5) Accumulate update 
   
    # Update the values 
    h2[kernel_times,:,:] += update2 / constant
    h1[kernel_times,:,:] += update1 / constant
    
    return h1, h2, Dh1, Dh2, Dg1, Dg2 


def learn_kernel_tolerance_adadelta(tolerance, training_indexes, delay_indexes, image_indexes, V, decay_rate, epsilon, kernel_times, 
                                    input_to_image, kernel_to_input, h0, h1, h2, ims, ims2, constant, verbose=False, modulus=10):
    '''
    Documentation here 
    '''
    
    # Initialize accumulate update and gradient 
    Dh1 = np.zeros(np.shape(h1))
    Dh2 = Dh1 
    Dg1 = Dh1 
    Dg2 = Dh1 
    
    # Initialize error 
    total_cum_error = training_indexes.size
    iteration = 0
    
    while((total_cum_error / training_indexes.size) > tolerance): #General loop
        
        iteration += 1  # Count the number of iterations 
        
        permutation = np.random.choice(training_indexes, training_indexes.size, replace=False) # Shuffle the data 
        
        for index in permutation: # Loop over training examples 
            # calculate the error 
            error = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                  kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            h1, h2, Dh1, Dh2, Dg1, Dg2 = update_kernel_adadelta(epsilon, decay_rate, error, index, input_to_image, kernel_to_input,
                            kernel_times, h0, h1, h2, ims, ims2, Dg1, Dg2, Dh1, Dh2, constant)
            #h0 = h0 + learning_rate * error
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, kernel_times, input_to_image, 
                                                kernel_to_input, h0, h1, h2, ims, ims2)      
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            
            if (verbose == True):
                # Print information regarding the progress of the algorithm
                print '*********************************'
                print 'iteration', iteration 
                print 'total cumulative error', total_cum_error
                print 'total cumulative error normalized', total_cum_error / training_indexes.size
                print 'total cumulative error change', total_cum_error - AUX
                print 'cumulative ratio', total_cum_error / AUX 
                print 'decay rate and epislon', decay_rate, epsilon
                print '*********************************'
    
    return h0, h1, h2 




def learn_kernel_tolerance_adaptative(tolerance, training_indexes, V, learning_rate, kernel_times,
                                      input_to_image, kernel_to_input, h0, h1, h2, ims, ims2, verbose=False, modulus=10):
    '''
    Documentation here 
    '''
    total_cum_error = training_indexes.size
    iteration = 0
    
    while((total_cum_error / training_indexes.size) > tolerance): #General loop
        
        iteration += 1  # Count the number of iterations 
        
        # Learning rate 
        #exponent = (np.log10(iteration) / 0.4) + 3.9
        #learning_rate = 10 ** (-exponent)
         
        permutation = np.random.choice(training_indexes, training_indexes.size, replace=False) # Shuffle the data 
        
        for index in permutation: # Loop over training examples 
            # calculate the error 
            error = measure_error(V, index, kernel_times, input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            h1, h2 = update_kernel(learning_rate, error, index, input_to_image, kernel_to_input, kernel_times, h0, h1, h2, ims, ims2)
            #h0 = h0 + learning_rate * error
         
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, kernel_times, input_to_image, 
                                                kernel_to_input, h0, h1, h2, ims, ims2)  
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            
            if (verbose == True):
                # Print information regarding the progress of the algorithm
                print '*********************************'
                print 'iteration', iteration 
                print 'total cumulative error', total_cum_error
                print 'total cumulative error normalized', total_cum_error / training_indexes.size
                print 'total cumulative error change', total_cum_error - AUX
                print 'cumulative ratio', total_cum_error / AUX 
                print 'learning_rate', learning_rate
                print '*********************************'
    
    return h0, h1, h2 




def learn_kernel_tolerance_mix(tolerance, training_indexes, V, decay_rate, epsilon, dt, dh, dim, 
                                    kernel_size, h0, h1, h2, ims, ims2, verbose=False, modulus=10):
    '''
    Documentation here 
    '''
    
    # Initialize accumulate update and gradient 
    Dh1 = np.zeros(np.shape(h1))
    Dh2 = Dh1 
    Dg1 = Dh1 
    Dg2 = Dh1 
    
    # Initialize error 
    total_cum_error = training_indexes.size
    iteration = 0
    
    kernel_times = np.arange(kernel_size)
    kernel_times = kernel_times.astype(int) # Make the values indexes 
    
    # Scale factors 
    input_to_image  = dt / dim # Transforms input to image  
    kernel_to_input = dh / dt  # Transforms kernel to input 
    
    while((total_cum_error / training_indexes.size) > tolerance): #General loop
        
        iteration += 1  # Count the number of iterations 
        print iteration 
        permutation = np.random.choice(training_indexes, training_indexes.size, replace=False) # Shuffle the data 
        
        for index in permutation: # Loop over training examples 
            # calculate the error 
            error = measure_error(V, index, kernel_times, input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            
            if (iteration < 10):
                h1, h2 = update_kernel(10**(-4.5), error, index, input_to_image, 
                                       kernel_to_input, kernel_times, h0, h1, h2, ims, ims2) 
            else:
                h1, h2, Dh1, Dh2, Dg1, Dg2 = update_kernel_adadelta(epsilon, decay_rate, error, index, input_to_image, kernel_to_input,
                                kernel_times, h0, h1, h2, ims, ims2, Dg1, Dg2, Dh1, Dh2)
            #h0 = h0 + learning_rate * error
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, dt, dh, dim, kernel_size, h0, h1, h2, ims, ims2)      
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            
            if (verbose == True):
                # Print information regarding the progress of the algorithm
                print '*********************************'
                print 'iteration', iteration 
                print 'total cumulative error', total_cum_error
                print 'total cumulative error normalized', total_cum_error / training_indexes.size
                print 'total cumulative error change', total_cum_error - AUX
                print 'cumulative ratio', total_cum_error / AUX 
                print 'decay rate and epislon', decay_rate, epsilon
                print '*********************************'
    
    return h0, h1, h2 


#######
# Linear learning 
#########

def update_kernel_linear(learning_rate, error, index, delay_indexes, image_indexes, input_to_image, kernel_to_input, kernel_times, h0, h1, h2, ims, ims2): 
    '''
    update kernel 
    
    '''
    delay = image_indexes[index - delay_indexes] 
       
    # Update the values 
    h1[kernel_times,:,:] += learning_rate * error * ims[delay,:,:]
    
        
    return h1, h2

def learn_kernel_tolerance_linear(tolerance, training_indexes, delay_indexes, image_indexes, V, learning_rate, kernel_times, 
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
        
        #for index in permutation: # Loop over training examples 
        for index in training_indexes:
            # calculate the error 
            error = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                  kernel_to_input, h0, h1, h2, ims, ims2)
            # Update the kernel with that error 
            h1 = update_kernel_linear(learning_rate, error, index, delay_indexes, image_indexes, input_to_image, 
                                   kernel_to_input, kernel_times, h0, h1, h2, ims, ims2)
            
            #h0 = h0 + learning_rate * error
         
    
        if (iteration % modulus == 0): 
            total_error = calculate_total_error(training_indexes, V, kernel_times, delay_indexes, image_indexes,
                                                 input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
            AUX = total_cum_error 
            total_cum_error = np.sum(total_error)                
            
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


#################
# Error functions
#################

def measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, kernel_to_input, h0, h1, h2, ims, ims2):
    '''
    Gives the difference between the value and the prediction 
    '''
    return (V[index] - hypothesis(index, kernel_times, delay_indexes, image_indexes, 
                                  input_to_image, kernel_to_input, h0, h1, h2, ims, ims2))


def calculate_total_error(data_indexes, V, kernel_times, delay_indexes, image_indexes, 
                          input_to_image, kernel_to_input, h0, h1, h2, ims, ims2):
    '''
    returns a vector whose entries are the square errors for every member of the data set    
    Here V is the array containing the full signal that we want to learn 
    data_indexes contains the particular indexes that we are interested in
    '''
    
    error = np.zeros(data_indexes.size)
     
    for index_index,index in enumerate(data_indexes):
        error[index_index] = measure_error(V, index, kernel_times, delay_indexes, image_indexes, input_to_image, 
                                           kernel_to_input, h0, h1, h2, ims, ims2)**2
    
    return error 


def calculate_prediction(data_indexes, kernel_times, delay_indexes, image_indexes, input_to_image, 
                         kernel_to_input, h0, h1, h2, ims, ims2):
    '''
    returns an array with the predictions for the indexes given in data_indexes
    '''
    prediction = np.zeros(data_indexes.size)
    
    # Scale factors     
    # Calculate prediction
    for index_index, index in enumerate(data_indexes):
        prediction[index_index] = hypothesis(index, kernel_times, delay_indexes, image_indexes, 
                                  input_to_image, kernel_to_input, h0, h1, h2, ims, ims2)
    
    return prediction






    
    
