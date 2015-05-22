#!/usr/bin/python

"""
Created on Wed Mar 25 13:23:56 2015

@author: t7
"""

#IMPORTS=======================================================================
#import sys
##mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
#sys.path.append(mypath)

from pandas import Series, DataFrame
from nilmtk import DataSet, HDFDataStore

from disaggregate.combinatorial_optimisation_location import CombinatorialOptimisation
from feature_detectors.location_inference import LocationInference
from metadata.metadata import Metadata
from stats.ground_truth import GroundTruth
from stats.metrics import Metrics
import stats.metrics as metrics
import settings as settings
import misc.utils as utils

from pylab import rcParams
import matplotlib.pyplot as plt
import time

start_time = time.time()
#CONSTANTS=====================================================================
rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')

#ADD NILM-Loc path to PYTHON PATH in IDE preferences

settings.akshay = True
if settings.akshay:
    root_path = settings.path_root_b
else:
    root_path = settings.path_root_a

settings.h5_redd = root_path + settings.h5_redd
settings.h5_eco  = root_path + settings.h5_eco
settings.h5_iawe = root_path + settings.h5_iawe

ds_path = root_path + '/Datasets/REDD/'
fn_path = ds_path + 'comparisons/'
fn_gt  = 'gt_values_20110422.csv'
fn_obj = 'c2fp'
fn_save = fn_path + fn_obj
save_files_and_objects = False

h5_disag               = ds_path + 'redd-disag-loc.h5'
#h5_disag_redd_original = ds_path + 'redd-disag-original.h5'
h5_disag_redd_original = ds_path + 'redd-disag-original-modified-centroids.h5'

dataset_name = 'REDD'
#dataset_start_date_disag = None
#dataset_end_date_disag   = '2011-04-19 00:00:00'
dataset_start_date_disag = '2011-04-18 00:00:00'
dataset_end_date_disag   = '2011-04-24 23:10:00'

manual_centroids = True
nclusters = 2
dataset_start_date_train = '2011-04-19 00:00:00'
dataset_end_date_train = '2011-04-22 00:00:00'


#ORIGINAL CO DISAGGREGATION====================================================
#Running from original source code
print("Original CO============================================================")
vampire_power_in_original = 90.81
#metaREDD = Metadata('REDD')
#plain_co, vampire_power_in_original = utils.disaggregate_original_co(settings.h5_redd, h5_disag_redd_original, centroids=metaREDD.centroids)

time_start_loc = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_loc - start_time))
print("Section Original CO   : %s seconds ---\n" % (time_start_loc - start_time))





#INFER LOCATIONS===============================================================
print("Inferring locations====================================================")
loc = LocationInference(dataset_name)
loc.dataset.set_window(start=dataset_start_date_disag, end=dataset_end_date_disag)
loc.infer_locations()

time_start_train = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_train - start_time))
print("Section Locations     : %s seconds ---\n" % (time_start_train - time_start_loc))



#TRAINING======================================================================
print("Training===============================================================")
co = CombinatorialOptimisation()
if manual_centroids:
    co.train(loc.elec, centroids=loc.metadata.centroids) #Set centroids manually
else:
    loc.dataset.set_window(start=dataset_start_date_train, end=dataset_end_date_train)
    co.train(loc.elec, max_num_clusters = nclusters) #Train centroids

time_start_gt = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_gt - start_time))
print("Section Training      : %s seconds ---\n" % (time_start_gt - time_start_train))



#CALCULATE GROUND TRUTH========================================================
print("Calculating ground truth===============================================")
loc.dataset.set_window(start=dataset_start_date_disag, end=dataset_end_date_disag)
gt = GroundTruth(loc, co, baseline=vampire_power_in_original)
gt.generate()

time_start_disag = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_disag - start_time))
print("Section Ground truth  : %s seconds ---\n" % (time_start_disag - time_start_gt))



#DISAGREGGATION================================================================
print("Disaggregating=========================================================")
output = HDFDataStore(h5_disag, 'w')
loc.dataset.set_window(start=dataset_start_date_disag, end=dataset_end_date_disag)
co.disaggregate(loc.elec.mains(), output, location_data=loc, baseline=vampire_power_in_original, resample_seconds=60)
output.close()

time_start_metrics = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_metrics - start_time))
print("Section Disaggregation: %s seconds ---\n" % (time_start_metrics - time_start_disag))



#METRICS=======================================================================
print("Calculating metrics====================================================")
disag  = DataSet(h5_disag)
disago = DataSet(h5_disag_redd_original)

disago.metadata['timezone'] = disag.metadata['timezone']
disago.set_window(start=dataset_start_date_disag, end=dataset_end_date_disag)

disag_elec  = disag.buildings[1].elec
disago_elec = disago.buildings[1].elec

disag_predictions_original = utils.get_disaggregation_predictions(disago_elec,
                                          vampire_power_in_original, 
                                          start_date = dataset_start_date_disag, 
                                          end_date = dataset_end_date_disag)
disag_predictions_location = utils.get_disaggregation_predictions(disag_elec,
                                          vampire_power_in_original, 
                                          start_date = dataset_start_date_disag, 
                                          end_date = dataset_end_date_disag)                                          
                                          
mt = Metrics(co, gt, loc, disag_elec, disago_elec)
mt.calculate()
mt.build_results_tables()

time_start_save = time.time()
print("\nTotal elapsed: %s seconds ---" % (time_start_save - start_time))
print("Section Metrics       : %s seconds ---\n" % (time_start_save - time_start_metrics))



#SAVE RESULTS==================================================================  
if save_files_and_objects:     
    print("Saving files and object================================================")
    mt.save_to_files(fn_save)
    mt.save_objects(fn_save)
    gt.save_to_file(fn_path + fn_gt)  


#Read objects
#r1_objects = mt.read_objects(fn_save)
#d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs = metrics.dismantle_object(r1_objects)
end_time = time.time()



print('Done!==================================================================')
print("\nTotal: %s seconds ---" % (end_time - start_time))
print("Section Original CO   : %s seconds ---" % (time_start_loc - start_time))
print("Section Locations     : %s seconds ---" % (time_start_train - time_start_loc))
print("Section Training      : %s seconds ---" % (time_start_gt - time_start_train))
print("Section Ground truth  : %s seconds ---" % (time_start_disag - time_start_gt))
print("Section Disaggregation: %s seconds ---" % (time_start_metrics - time_start_disag))
print("Section Metrics       : %s seconds ---" % (time_start_save - time_start_metrics))
print("Section Save results  : %s seconds ---" % (end_time - time_start_save))