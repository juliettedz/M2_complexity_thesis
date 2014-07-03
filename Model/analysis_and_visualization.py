import numpy
import mozaik
import pylab
import cPickle
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.circ_stat import circular_dist
from mozaik.tools.mozaik_parametrized import MozaikParametrized

def perform_analysis_and_visualization(data_store):
    analog_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_esyn_ids()
    analog_ids_inh = param_filter_query(data_store,sheet_name="V1_Inh_L4").get_segments()[0].get_stored_esyn_ids()
    spike_ids = param_filter_query(data_store,sheet_name="V1_Exc_L4").get_segments()[0].get_stored_spike_train_ids()
    
    number_of_cells = len(analog_ids)
    stimuli_list = list(('SparseNoise', 'DenseNoise'))
    save_to = './Data/'
    
    
    for stimuli_type in stimuli_list: 
        print 'Getting voltage and images for ' + stimuli_type
        
        # Saving parameters 
        format = '.pickle'
        quality = '_3000_21_' # This is the number of images followed by the interval that they take in ms 
        
        # Load the segments 
        dsv = queries.param_filter_query(data_store, sheet_name="V1_Exc_L4",st_name=stimuli_type)
        segments = dsv.get_segments()
        stimuli = [MozaikParametrized.idd(s) for s in dsv.get_stimuli()]
        # Take the seeds 
        seeds = [s.experiment_seed for s in stimuli]
                
        # Sort them based on their seeds 
        seeds,segments,stimuli = zip(*sorted(zip(seeds,segments,stimuli))) 
        segment_length = segments[0].get_spiketrain(spike_ids[0]).t_stop     
        
        # Values to obtain 
        spikes = [[] for i in segments[0].get_spiketrain(spike_ids)]    
        images = []
        
        ## Extract images 
        print 'Extracting and processing images'
        for i, seg in enumerate(segments):
            """
            First we take out the stimuli and make them as small as we can First we take out the stimuli and make them 
            as small as we can than one pixel assigned to each value of luminance. In order to do so, we first call the class
            And re-adjust is parameter st.density = st.grid_size. After that we successively call the class to extract the images
            frames  
            """
                        
            # First we take the class 
            st = MozaikParametrized.idd_to_instance(stimuli[i])
            st.size_x = 1.0
            st.size_y = 1.0 
            st.density = st.grid_size
            
            fr = st.frames()          
            
            # First we call as many frames as many frames as we need (total time / time per image = total # of images) 
            ims = [fr.next()[0] for i in xrange(0,st.duration/st.frame_duration)]
            # Now, we take the images that repeat themselves 
            ims = [ims[i] for i in xrange(0,len(ims)) if ((i % (st.time_per_image/st.frame_duration)) == 0)] 
            images.append(ims)
        
        # Saving images 
        print 'Saving Images '
        
        # Concatenate and save 
        ims = numpy.concatenate(images,axis=0)  
        images_filename = save_to + 'images' + quality + stimuli_type + format
        f = open(images_filename,'wb')
        cPickle.dump(ims,f)
        
        ## Get the voltage for all the cells 
        for cell_number in range(number_of_cells):
            print 'Extracting Voltage for cell ', cell_number 
            
            vm = [] # Intialize voltage list 
            for i,seg in enumerate(segments):
                # get vm 
                v = seg.get_vm(analog_ids[cell_number])
                # Check that the voltage between segments match
                if vm != []:
                    assert vm[-1][-1] == v.magnitude[0]                 
                # Append 
                vm.append(v.magnitude)    
                
              
            # Concatenate the experiments
            print 'Concatenating Voltage'
            vm = [v[:-1] for v in vm] # Take the last element out  
            vm = numpy.concatenate(vm,axis=0)
            
            print 'voltage shape=', numpy.shape(vm)
            # Save the voltage
            print 'Saving Voltage for cell', cell_number 
            voltage_filename = save_to + 'vm' + '_cell_'+ str(cell_number) + quality + stimuli_type + '.pickle'
            f = open(voltage_filename,'wb')
            cPickle.dump(vm,f)         
          
        
        ## Spikes 
        #for i,seg in enumerate(segments):
            # get spikes    
            #sp = seg.get_spiketrain(spike_ids)
            #sp = [list(s.magnitude+i*segment_length.magnitude) for s in sp]
            
            #spikes = [a+b for a,b in zip(spikes,sp)]
            #f = open('spikes.pickle','wb')
            #cPickle.dump(spikes,f)