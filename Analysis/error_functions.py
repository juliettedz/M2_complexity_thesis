import numpy as np
from learning_functions import hypothesis 

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



