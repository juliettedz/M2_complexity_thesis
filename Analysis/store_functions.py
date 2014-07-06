## Store functions 

import h5py
import numpy as np
from time import localtime

def store_kernel_hdf(dt, dh, dim, kernel_size, h0, h1, h2, Ntraining, iterations, learning_rate):
    # Extract values 
    Nside = np.shape(h1)[1]
    
    # filename
    directory = '../data/'
    date_stamp = '%4d-%02d-%02dT%02d-%02d-%02d' % localtime()[:6]
    format_stamp ='.hdf5'
    title = 'kernels' 
    filename = directory + date_stamp + title + format_stamp
    
    # STore the files 
    f = h5py.File(filename,'a')
    f.create_dataset('h0', data = h0, dtype=np.float32)
    h1 = f.create_dataset('h1', data = h1, dtype=np.float32)
    h2 = f.create_dataset('h2', data = h2, dtype=np.float32)
    # Save metadata
    h1.attrs['dt'] = dt
    h1.attrs['dh'] = dh
    h1.attrs['dim'] = dim
    h1.attrs['kernel_size'] = kernel_size
    h1.attrs['Ntraining'] = Ntraining
    h1.attrs['iterations'] = iterations
    h1.attrs['learning_rate'] = learning_rate


def store_kernel_numpy(kernel_size, h1, h2, cell_number, stimuli_type):
    # filename
    directory = './kernels/'
    
    parameter_stamp = str(cell_number) + stimuli_type
    format_stamp = '.npy'
    title1 = 'h1'
    title2 = 'h2'
    filename1 = directory  + title1 + parameter_stamp + format_stamp
    filename2 = directory  + title2 + parameter_stamp + format_stamp

    np.save(filename1, h1)
    np.save(filename2, h2)

def store_kernel_numpy_parameters(dt, dh, dim, kernel_size, h1, h2, Ntraining, string):
    '''
    Save on of the filters calculate with exact least squares 
    '''
    # filename
    directory = './kernels/'
    date_stamp = '%4d-%02d-%02dT%02d-%02d-%02d' % localtime()[:6]
    parameter_stamp = 'dt='+str(dt)+'dh='+str(dh)+'dim='+str(dim)+'kernel_size='+str(kernel_size)+'Ntraining=' + str(Ntraining)
    format_stamp = '.npy'
    title1 = 'h1' + string
    title2 = 'h2' + string
    filename1 = directory + date_stamp + title1 + parameter_stamp + format_stamp
    filename2 = directory + title2 + date_stamp + parameter_stamp + format_stamp

    np.save(filename1, h1)
    np.save(filename2, h2)
    
def store_X(X, tau, string):
    '''
    Store the X matrix in the least squares calculations to calculate the covariance matrix 
    '''
    
    directory = './X_matrices/'
    parameter_stamp = str(tau)
    format_stamp = '.npy'
    title = string + 'X'
    filename = directory + title + parameter_stamp + format_stamp
    np.save(filename, X)