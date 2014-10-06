"""
Contains functions to process the data 
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy import nanmean
from time import localtime
from math import floor

def sta_v(V, ims, training_indexes, delay_indexes, image_indexes, kernel_to_input, input_to_image, kernel_times, verbose=True):
    
    kernel_size = kernel_times.size
    Nside = np.shape(ims)[2]
    sta = np.zeros((kernel_size ,Nside, Nside))
    
    for tau, delay_index in zip(kernel_times, delay_indexes):
        if (verbose == True): # Print the delays 
            print 'tau=', tau
      
        delay = image_indexes[training_indexes - delay_index] # Take the image indexes 
        weighted_stimuli = np.sum( V[training_indexes, np.newaxis, np.newaxis] * ims[delay,...], axis=0)
        sta[tau,...] = weighted_stimuli / training_indexes.size
    
    #return np.transpose(sta, (0,2,1))
    return sta



def positive_time_trace(x, y , time_window, image_to_input, V, ims):

    # Identify the positive value  
    positive_value  = np.max(ims)

    # Take the indexes for which the images are positive 
    positive_bolean = ims[:,x,y] == positive_value # Take positive entries 
    positive_index = np.where(positive_bolean) 
    positive_index = positive_index[0] * image_to_input # Transform to input coordinates 
    positive_index = positive_index.astype(int)# Make them indexes 
        
    # I need to discard the last images
    positive_index = positive_index[ positive_index + time_window < V.size]
    
    # Initialize the traces  
    positive_time_trace = np.zeros(time_window) # Create the vector where the trace will be stored 
    
    # Get the traces
    for index in positive_index:
        positive_time_trace += V[index: index + time_window]
        
    positive_time_trace = positive_time_trace / positive_index.size

    return positive_time_trace

def negative_time_trace(x, y , time_window, image_to_input, V, ims):

    # Calculate the negative value 
    negative_value  = np.min(ims)

    # Take the indexes for which the have a negative value 
    negative_bolean = ims[:,x,y] == negative_value
    negative_index = np.where(negative_bolean)
    negative_index = negative_index[0] * image_to_input
    negative_index = negative_index.astype(int)
    
    # Discard last images 
    negative_index = negative_index[ negative_index + time_window < V.size]
    
    # Initialize the traces  
    negative_time_trace = np.zeros(time_window)
    
    # Get the traces    
    for index in negative_index:
        negative_time_trace += V[index: index + time_window]
        

    negative_time_trace = negative_time_trace / negative_index.size

    return negative_time_trace


## Scale and sampling 

def downsample(signal, factor):
    # fill with NaN till the size is divisible by the factor
    pad_size = np.ceil(float(signal.size)/factor)*factor - signal.size
    pad_size = np.int(pad_size)
    b_padded = np.append(signal, np.zeros(pad_size)*np.NaN)
    # Reshape by the factor and take the mean 
    factor = np.int(factor)
    return nanmean(b_padded.reshape(-1,factor), axis=1)

    
def z_score(h):
    hmean = np.mean(h, axis=0)
    hstd = np.std(h, axis=0)
    hz = (h - hmean) / hstd
    return hz

def z_score2(h):
    hmean = np.mean(h)
    hstd = np.std(h)
    hz = (h - hmean) / hstd 
    return hz 
    
