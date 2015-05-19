#!/usr/bin/python

"""
Created on Wed Mar 25 13:23:56 2015

@author: t7
"""

#IMPORTS=======================================================================
import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
sys.path.append(mypath)
import settings as settings

from pandas import Series, DataFrame
from nilmtk import DataSet, HDFDataStore

from disaggregate.combinatorial_optimisation_location import CombinatorialOptimisation
from feature_detectors.location_inference import LocationInference
from stats.ground_truth import GroundTruth
from stats.metrics import Metrics
import stats.metrics as metrics

from pylab import rcParams
import matplotlib.pyplot as plt
import time

start_time = time.time()
#CONSTANTS=====================================================================
rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')

ds_path = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/iAWE/'

fn_path = ds_path + 'comparisons/'
fn_gt  = 'gt_values_xxxxxxxx.csv'
fn_obj = 'r1'
fn_save = fn_path + fn_obj
save_files_and_objects = False

h5_disag               = ds_path + 'redd-disag-loc.h5'
h5_disag_redd_original = ds_path + 'redd-disag-original-modified-centroids.h5'

dataset_name = 'iAWE'
dataset_start_date = None
dataset_end_date = '2011-04-19 00:00:00'


iawe = DataSet(settings.h5_iawe)
#TimeFrame(start='2013-05-24 05:30:00+05:30', end='2013-09-18 08:40:55+05:30', empty=False)
iawe.set_window(start='2011-04-19 00:00:00', end='2011-04-27 00:00:00')
elec = iawe.buildings[1].elec











#time_start_loc = time.time()
##INFER LOCATIONS===============================================================
#print("Inferring locations====================================================")
#loc = LocationInference(dataset_name)
#loc.dataset.set_window(start=dataset_start_date, end=dataset_end_date)
#loc.infer_locations()
#
#time_start_train = time.time()
#print("\nTotal elapsed: %s seconds ---" % (time_start_train - start_time))
#print("Section Locations     : %s seconds ---\n" % (time_start_train - time_start_loc))
#
#
#
##TRAINING======================================================================
#print("Training===============================================================")
#co = CombinatorialOptimisation()
#co.train(loc.elec, centroids=loc.metadata.centroids) #Set centroids manually
#
#time_start_gt = time.time()
#print("\nTotal elapsed: %s seconds ---" % (time_start_gt - start_time))
#print("Section Training      : %s seconds ---\n" % (time_start_gt - time_start_train))
#
#
#
##CALCULATE GROUND TRUTH========================================================
#print("Calculating ground truth===============================================")
#gt = GroundTruth(loc, co)
#gt.generate()
#
#time_start_disag = time.time()
#print("\nTotal elapsed: %s seconds ---" % (time_start_disag - start_time))
#print("Section Ground truth  : %s seconds ---\n" % (time_start_disag - time_start_gt))
#
#
#
##DISAGREGGATION================================================================
#print("Disaggregating=========================================================")
#output = HDFDataStore(h5_disag, 'w')
#co.disaggregate(loc.elec.mains(), output, location_data=loc, resample_seconds=60)
#output.close()
#
#time_start_metrics = time.time()
#print("\nTotal elapsed: %s seconds ---" % (time_start_metrics - start_time))
#print("Section Disaggregation: %s seconds ---\n" % (time_start_metrics - time_start_disag))
#
#
#
##METRICS=======================================================================
#print("Calculating metrics====================================================")
#disag  = DataSet(h5_disag)
#disago = DataSet(h5_disag_redd_original)
#disag_elec  = disag.buildings[1].elec
#disago_elec = disago.buildings[1].elec
#
#mt = Metrics(co, gt, loc, disag_elec, disago_elec)
#mt.calculate()
#mt.build_results_tables()
#
#time_start_save = time.time()
#print("\nTotal elapsed: %s seconds ---" % (time_start_save - start_time))
#print("Section Metrics       : %s seconds ---\n" % (time_start_save - time_start_metrics))
#
#
#
##SAVE RESULTS==================================================================  
#if save_files_and_objects:     
#    print("Saving files and object================================================")
#    mt.save_to_files(fn_save)
#    mt.save_objects(fn_save)
#    gt.save_to_file(fn_path + fn_gt)  
#
#
##Read objects
##r1_objects = mt.read_objects(fn_save)
##d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs = metrics.dismantle_object(r1_objects)
#end_time = time.time()
#
#
#
#print('Done!==================================================================')
#print("\nTotal: %s seconds ---" % (end_time - start_time))
#print("Section Locations     : %s seconds ---" % (time_start_train - time_start_loc))
#print("Section Training      : %s seconds ---" % (time_start_gt - time_start_train))
#print("Section Ground truth  : %s seconds ---" % (time_start_disag - time_start_gt))
#print("Section Disaggregation: %s seconds ---" % (time_start_metrics - time_start_disag))
#print("Section Metrics       : %s seconds ---" % (time_start_save - time_start_metrics))
#print("Section Save results  : %s seconds ---" % (end_time - time_start_save))