#!/usr/bin/python

"""
Created on Wed Mar 25 13:23:56 2015

@author: t7
"""

#IMPORTS=======================================================================
import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
sys.path.append(mypath)

from pandas import Series, DataFrame
from nilmtk import DataSet, HDFDataStore

from disaggregate.combinatorial_optimisation_location import CombinatorialOptimisation
from feature_detectors.location_inference import LocationInference
from stats.ground_truth import GroundTruth
from stats.metrics import Metrics
import stats.metrics as metrics

from pylab import rcParams
import matplotlib.pyplot as plt

#CONSTANTS=====================================================================
rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')

ds_path = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/'

fn_path = ds_path + 'comparisons/'
fn_gt  = 'gt_values_20110418.csv'
fn_obj = 'r1'
fn_save = fn_path + fn_obj
save_files_and_objects = False

h5_disag               = ds_path + 'redd-disag-loc.h5'
h5_disag_redd_original = ds_path + 'redd-disag-original-modified-centroids.h5'

dataset_name = 'REDD'
dataset_start_date = None
dataset_end_date = '2011-04-19 00:00:00'


#INFER LOCATIONS===============================================================
print("Inferring locations...")
loc = LocationInference(dataset_name)
loc.dataset.set_window(start=dataset_start_date, end=dataset_end_date)
loc.infer_locations()

#TRAINING======================================================================
print("Training...")
co = CombinatorialOptimisation()
co.train(loc.elec, centroids=loc.metadata.centroids) #Set centroids manually


#CALCULATE GROUND TRUTH========================================================
print("Calculating ground truth...")
gt = GroundTruth(loc, co)
gt.generate()

#DISAGREGGATION================================================================
print("Disaggregating...")
output = HDFDataStore(h5_disag, 'w')
co.disaggregate(loc.elec.mains(), output, location_data=loc, resample_seconds=60)
output.close()


#METRICS=======================================================================
print("Calculating metrics...")
disag  = DataSet(h5_disag)
disago = DataSet(h5_disag_redd_original)
disag_elec  = disag.buildings[1].elec
disago_elec = disago.buildings[1].elec

mt = Metrics(co, gt, loc, disag_elec, disago_elec)
mt.calculate()
mt.build_results_tables()


#SAVE RESULTS==================================================================  
if save_files_and_objects:     
    print("Saving files and object...")
    mt.save_to_files(fn_save)
    mt.save_objects(fn_save)
    gt.save_to_file(fn_path + fn_gt)  


#Read objects
#r1_objects = mt.read_objects(fn_save)
#d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs = metrics.dismantle_object(r1_objects)

print('Done!')

