# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:23:56 2015

@author: t7
"""
from pandas import Series, DataFrame
from nilmtk import DataSet
from nilmtk.metrics import f1_score
import numpy as np
from nilmtk import HDFDataStore
import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/nilmtk'
sys.path.append(mypath)

from combinatorial_optimisation_location import CombinatorialOptimisation
from location_inference import LocationInference
import metrics as metrics

from pylab import rcParams
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')

#TRAINING======================================================================
print("Training...")

loc = LocationInference('REDD')
#loc.dataset.set_window(start='2011-04-19 00:00:00', end='2011-04-27 00:00:00')
#loc.infer_locations()

co = CombinatorialOptimisation()
co.train(loc.elec, max_num_clusters = 2, resample_seconds=60)

#Modify centroids manually
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

#DISAGREGGATION================================================================
print("Disagreggating...")
loc.dataset.set_window(start=None, end='2011-04-19 00:00:00')#(start='2011-05-16 00:00:00', end='2011-05-17 00:00:00')
loc.infer_locations()
mains = loc.elec.mains()

#GT values
vampire_power, centroids, state_combinations, summed_power_of_each_combination = metrics.generate_state_combinations_all(co, mains)
chunks = metrics.generate_mains_power_series(mains)
ps, pst     = metrics.generate_apps_power_series(loc)
gt_ps = DataFrame(ps)
gt_pst = DataFrame(pst)

correction = 0#28.35
locations_lists, appliances_lists, timestamps_list, mains_values, gt, gt_sums, gt_residuals, gt_states = metrics.get_gt_values(chunks, loc, (vampire_power+correction), ps, co)
t, mains_from_apps, smains, diffs, gt_all = metrics.compare_mains_and_gt_of_appliances(ps, mains_values, timestamps_list, (vampire_power+correction), gt_pst)
gt_sums[0] = gt_sums[1]
mains_from_apps = mains_from_apps - correction
gt_residuals = metrics.get_difference(mains_from_apps,gt_sums)

disagout  = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-loc.h5'
disagouto = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/redd-disag-original-modified-centroids.h5'
output = HDFDataStore(disagout, 'w')
#co.disaggregate(mains, output, location_data=loc, mains_values=mains_from_apps, resample_seconds=60)
co.disaggregate(mains, output, location_data=loc, resample_seconds=60)
output.close()


#METRICS=======================================================================
print("Metrics...")
disag = DataSet(disagout)
disag_elec  = disag.buildings[1].elec

disago = DataSet(disagouto)
disago_elec = disago.buildings[1].elec

#F1 score
f1_original = f1_score(disago_elec, loc.elec)
f1_loc = f1_score(disag_elec, loc.elec)
diff1  = [f1_loc.values[i] - f1_original.values[i] for i,v in enumerate(f1_loc)]
f1_diff_loc_vs_original = Series(diff1, index=f1_loc.index)

#Fraction assigned correctly
fraction_original = metrics.fraction_energy_assigned_correctly(disago_elec, loc.elec)
fraction_loc      = metrics.fraction_energy_assigned_correctly(disag_elec, loc.elec)




#Sums of combos and corresponding residuals
summed_power_of_each_combination = summed_power_of_each_combination + correction

co_original_combo_sums, co_location_combo_sums     = metrics.get_summed_power_of_combos(co, summed_power_of_each_combination)
co_residuals_original, co_residuals_location       = metrics.get_residuals_of_combos(co)
gt_sum_res, co_sum_res, loc_sum_res                = metrics.create_sum_residual_tuples(gt_sums, gt_residuals, co_original_combo_sums, co_residuals_original, co_location_combo_sums, co_residuals_location)

#Appliances and states guessed
co_original_combos, co_location_combos             = metrics.get_appliances_in_combos(co, loc, state_combinations)
co_original_states, co_location_states             = metrics.get_states_of_appliances_in_combos(co, vampire_power, state_combinations)
gt_combo_states, co_combo_states, loc_combo_states = metrics.create_app_state_tuples(gt, gt_states, co_original_combos, co_original_states, co_location_combos, co_location_states)

#Jaccard
jacc_co, jacc_loc, jacc_co_states, jacc_loc_states = metrics.jaccard2(gt_combo_states, co_combo_states, loc_combo_states)
jacc_co_apps_states, jacc_loc_apps_states          = metrics.create_jaccard_apps_states_tuples(jacc_co, jacc_loc, jacc_co_states, jacc_loc_states)
jaccard_results                                    = metrics.jaccard_total(jacc_co, jacc_loc, jacc_co_states, jacc_loc_states)



#Metrics paper=================================================================
pr_co = metrics.get_predicted_values_from_combos_found(loc, co_combo_states)
pr_loc = metrics.get_predicted_values_from_combos_found(loc, loc_combo_states)

spr_co  = metrics.predicted_values_to_series(pr_co, timestamps_list)
spr_loc = metrics.predicted_values_to_series(pr_loc, timestamps_list)
dis_co = DataFrame(spr_co)
dis_loc = DataFrame(spr_loc)

#Proportion Error per appliance
#proportion_error_co,  gt_proportion_co,  pr_proportion_co   = metrics.proportion_error_per_appliance(loc, mains_values, pst, spr_co)
#proportion_error_loc, gt_proportion_loc, pr_proportion_loc  = metrics.proportion_error_per_appliance(loc, mains_values, pst, spr_loc)
proportion_error_co,  gt_proportion_co,  pr_proportion_co   = metrics.proportion_error_per_appliance_df(smains, gt_pst, dis_co)
proportion_error_loc, gt_proportion_loc, pr_proportion_loc  = metrics.proportion_error_per_appliance_df(smains, gt_pst, dis_loc)

#Normal Disaggregation Error per appliance
#normal_error_co,  squares_of_diffs_co,  squares_of_gts_co  = metrics.normal_disaggregation_error_per_appliance(loc, pst, spr_co)
#normal_error_loc, squares_of_diffs_loc, squares_of_gts_loc = metrics.normal_disaggregation_error_per_appliance(loc, pst, spr_loc)
normal_error_co, sqrs_co   = metrics.normal_disaggregation_error_per_appliance_df(gt_pst, dis_co)
normal_error_loc, sqrs_loc = metrics.normal_disaggregation_error_per_appliance_df(gt_pst, dis_loc)

#Total Disaggregation Error
#total_error_co  = metrics.total_disaggregation_error(loc, squares_of_diffs_co, squares_of_gts_co)
#total_error_loc = metrics.total_disaggregation_error(loc, squares_of_diffs_loc, squares_of_gts_loc)
total_error_co  = metrics.total_disaggregation_error_df(gt_pst, dis_co)
total_error_loc = metrics.total_disaggregation_error_df(gt_pst, dis_loc)



#RESULTS=======================================================================

#Build dataframes to show results more clearly
d = metrics.build_results_table(locations_lists, appliances_lists, 
                        gt_combo_states, co_combo_states, loc_combo_states,
                        #mains_values, gt_sum_res, co_sum_res, loc_sum_res,
                        list(mains_from_apps.values), gt_sum_res, co_sum_res, loc_sum_res,
                        jacc_co_apps_states, jacc_loc_apps_states,
                        timestamps_list)

m, ma = metrics.build_metrics_tables(fraction_original, fraction_loc, jaccard_results, total_error_co, total_error_loc,
                         proportion_error_co, proportion_error_loc, normal_error_co, normal_error_loc,
                         gt_proportion_co, pr_proportion_co, pr_proportion_loc)
                         


#SAVE RESULTS==================================================================
gg = DataFrame(gt_pst)
del gg['diff1']
del gg['diff2']
gg['Loc Events'] = loc.events_apps_1min['Apps']
apps = loc.appliances_location.keys()
appst = loc.appliances_location.keys()
appst.remove(3)
appst.remove(4)
appst.remove(10)
appst.remove(20)
appst.append((3,4))
appst.append((10,20))
sd = {}
for app in apps:
    sd[app] = Series(0, index=gg.index)
for index, row in gg.iterrows():
    try:
        if len(row['Loc Events']) > 0:
            for app in apps:
                n = row['Loc Events'].count(app)
                sd[app][index] = n
    except Exception:
        continue
sd[(3,4)] = sd[3]
sd[(10,20)] = sd[10]
del sd[3]
del sd[4]
del sd[10]
del sd[20]
locevents = DataFrame(sd)
locevents.columns = [(str(col) + ' locEv') for col in locevents]
locevents.head()
for locEv in locevents:
    gg[locEv] = locevents[locEv]
act = DataFrame(loc.appliances_consuming_times)
act = act.resample('1Min')
del act[3]
del act[10]
act.columns = [(3,4), 5,6,7,8,9,11,12,13,14,15,16,17,18,19,(10,20)]
act.columns = [(str(col) + ' conEv') for col in act]
for app in act:
    gg[app] = act[app]
fnpath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/REDD/comparisons/gt_values_20110418'
gg.columns = [str(col) for col in gg]
gg = gg[sorted(gg.columns)]
gg.to_csv(fnpath+'.csv')    
   
#Save to files
#
#metrics.save_to_files(fnpath, d, m, ma)
#
###Save objects
#metrics.save_objects(fnpath, d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs)

##Read objects
#r1_objects = metrics.read_objects(fnpath)
#d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs = metrics.dismantle_object(r1_objects)