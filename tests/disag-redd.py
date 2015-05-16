# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:43:17 2015

@author: t7
"""

from nilmtk.dataset_converters import download_dataport
from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
import nilmtk
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.metrics import f1_score
from nilmtk import HDFDataStore
import numpy as np

source = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/low_freq'
outputhdf5 = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd.h5'
disagout = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-original-modified-centroids.h5'

print("Converting...")
#convert_redd(source, outputhdf5)
redd = DataSet(outputhdf5)

print("Training...")
#redd.set_window(start=None, end='2011-05-01 00:00:00')
redd.set_window(start='2011-04-19 00:00:00', end='2011-04-27 00:00:00')

elec = redd.buildings[1].elec
co = CombinatorialOptimisation()
co.train(elec)


co.model[0]['states'] = np.array([0, 49, 198])
co.model[1]['states'] = np.array([0, 1075])
co.model[2]['states'] = np.array([14,21])
co.model[3]['states'] = np.array([22,49,78])
co.model[4]['states'] = np.array([0,41,82])
co.model[5]['states'] = np.array([0,1520])
co.model[6]['states'] = np.array([0,1620])
co.model[7]['states'] = np.array([0,15])
co.model[8]['states'] = np.array([0,1450])
co.model[9]['states'] = np.array([0,1050])
co.model[10]['states'] = np.array([0,1450])
co.model[11]['states'] = np.array([0,65])
co.model[12]['states'] = np.array([0,45,60])
co.model[14]['states'] = np.array([0,4200])
co.model[15]['states'] = np.array([0,3750])



print("Disagreggating...")
#redd.set_window(start='2011-05-01 00:00:00', end=None)
redd.set_window(start=None, end='2011-04-19 00:00:00')
mains = elec.mains()
output = HDFDataStore(disagout, 'w')
co.disaggregate(mains, output)
output.close()

print("Metrics...")
disag = DataSet(disagout)
disag_elec = disag.buildings[1].elec
f1 = f1_score(disag_elec, elec)


#download_dataport('Aeo7SFRDqkRv', 
#                  'BIBoegKX88H6', 
#                  'wikienergy.h5',
#                  periods_to_load = {434: ('2014-04-01', '2014-05-01')})



#import sys
#mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/nilmtk'
#sys.path.append(mypath)
#from location_inference import LocationInference
#loc = LocationInference('REDD')
#loc.dataset.set_window(start='2011-04-19 00:00:00', end='2011-04-27 00:00:00')
#
#onTimes = loc.calculate_ON_times()
#events = loc.calculate_events()
#triggers = loc.calculate_triggers()
#locations = loc.calculate_locations()
#users = loc.calculate_users()