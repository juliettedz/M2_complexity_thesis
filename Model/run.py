# -*- coding: utf-8 -*-
"""
This is implementation of model of push-pull connectvity: 
Jens Kremkow: Correlating Excitation and Inhibition in Visual Cortical Circuits: Functional Consequences and Biological Feasibility. PhD Thesis, 2009.
"""
import sys
from pyNN import nest
from mozaik.controller import run_workflow, setup_logging
import mozaik
from model import PushPullCCModel
from experiments import create_experiments
from mozaik.storage.datastore import Hdf5DataStore,PickledDataStore
from analysis_and_visualization import perform_analysis_and_visualization
from parameters import ParameterSet


try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0



logger = mozaik.getMozaikLogger()

# Ju # put False if we already have all the date given by the model
if True:
    data_store,model = run_workflow('FFI',PushPullCCModel,create_experiments)
    #model.connectors['V1L4ExcL4ExcConnection'].store_connections(data_store)    
    #model.connectors['V1L4ExcL4InhConnection'].store_connections(data_store)    
    #model.connectors['V1L4InhL4ExcConnection'].store_connections(data_store)    
    #model.connectors['V1L4InhL4InhConnection'].store_connections(data_store)    
    #model.connectors['V1AffConnection'].store_connections(data_store)    
    #model.connectors['V1AffInhConnection'].store_connections(data_store)    
    
else: 
    setup_logging()
    data_store = PickledDataStore(load=True,parameters=ParameterSet
# Ju # change the name of the test you want to transform into data
({'root_directory':'FFI_test3_smallVF_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')

if mpi_comm.rank == MPI_ROOT:
    perform_analysis_and_visualization(data_store)
