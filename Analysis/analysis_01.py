################################################################################
## The kernels files for cells are read in this files and the SI  
## and the gains are calculated and plotted 
################################################################################

import numpy as np
import cPickle
import matplotlib.pyplot as plt
import os 

## Files 
images_folder = './data/'
kernels_folder = './kernels/'
quality = '_15000_21_' # _NumOfImages_NumOfCells
stimuli_type_sparse = 'SparseNoise'
stimuli_type_dense = 'DenseNoise'
image_format = '.pickle'
kernel_format = '.npy'


number_of_cells = 1
SI_sparse = np.zeros(number_of_cells)
SI_dense = np.zeros(number_of_cells)
gain1 = np.zeros(number_of_cells)
gain2 = np.zeros(number_of_cells)

for cell_number in xrange(number_of_cells):
    cell = '_cell_' + str(cell_number) 
    print 'cell', cell
    
    #### Load the sparse part 
    filename_images_sparse = images_folder + 'images' + quality + stimuli_type_sparse + image_format
    filename_h1kernel_sparse = kernels_folder + 'h1' + str(cell_number) + stimuli_type_sparse + kernel_format
    filename_h2kernel_sparse = kernels_folder + 'h2' + str(cell_number) + stimuli_type_sparse + kernel_format

    h1_sparse = np.load(filename_h1kernel_sparse)
    h2_sparse = np.load(filename_h2kernel_sparse)
    
    ## Load images
    f = open(filename_images_sparse,'rb' )
    ims_sparse = cPickle.load(f)
    f.close()
    
    #### Load the dense part 
    
    filename_images_dense = images_folder + 'images' + quality + stimuli_type_dense + image_format
    filename_h1kernel_dense = kernels_folder + 'h1' + str(cell_number) + stimuli_type_dense + kernel_format
    filename_h2kernel_dense = kernels_folder + 'h2' + str(cell_number) + stimuli_type_dense + kernel_format

    h1_dense = np.load(filename_h1kernel_dense)
    h2_dense = np.load(filename_h2kernel_dense)
    
    f = open(filename_images_dense,'rb' )
    ims_dense = cPickle.load(f)
    f.close()
    
    
    ###  Calculate the SI's 
    SI_sparse[cell_number] = np.sum(h1_sparse**2) / np.sum( h2_sparse**2 + h1_sparse**2)
    SI_dense[cell_number] = np.sum(h1_dense**2 ) / np.sum( h2_dense**2 + h1_dense**2)
    
    
    ## Calculate gains 
    gain1[cell_number] = np.sqrt(np.sum(h1_sparse**2)/np.sum(h1_dense**2))
    gain2[cell_number] = np.sqrt(np.sum(h2_sparse**2)/np.sum(h2_dense**2))
    


### ### ### 
# Plot and save 
### ### ### 

# Plot the gains 

plt.plot(gain1, gain2, '*')
plt.xlabel('h1 gain SN/DN')
plt.ylabel('h2 gain SN/DN ')
plt.xlim([0,20])
plt.ylim([0,20])

# Identity 
t = np.linspace(0,20,100)
plt.plot(t,t, 'k')


# Fit a linear function 
from scipy.optimize import curve_fit
def fit_func(x, a):
    return a*x 

params = curve_fit(fit_func, gain1, gain2)
k = params[0][0]
y = fit_func(gain1, k)

plt.plot(t, k*t, 'r')

# Change the font to font size
fontsize = 25
figure = plt.gcf()
ax = figure.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)


directory = './figures/'
formating='.pdf'
title = 'gain'
save_filename = directory + title + formating 

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 16)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))

plt.show()

# Plot the Simpleness Index 

plt.plot(SI_sparse, SI_dense, '*')
plt.xlabel('SI SN')
plt.ylabel('SI DN')
plt.xlim([0,1])
plt.ylim([0,1])

# Identity 
t = np.linspace(0,1,100)
plt.plot(t,t, 'k')

## Ploft the fit

alpha = (1.0 / k) ** 2

y = t / (t + alpha * (1 - t))
plt.plot(t ,y, 'r')


# Change the font to font size
figure = plt.gcf()
ax = figure.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)

directory = './figures/'
formating='.pdf'
title = 'SI'
save_filename = directory + title + formating 

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 16)
plt.savefig(save_filename, dpi = 100)
os.system("pdfcrop %s %s" % (save_filename, save_filename))
plt.show()



