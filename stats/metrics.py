# -*- coding: utf-8 -*-
from nilmtk.metergroup import MeterGroup
from pandas import DataFrame, Series, DateOffset
import numpy as np 
import cPickle as pickle
from math import sqrt
   
def fraction_energy_assigned_correctly(predictions, ground_truth):    
    predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)
    
    fraction_per_meter_predictions  = predictions_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()
    
    fraction = 0
    for meter_instance,v in enumerate(predictions_submeters.instance()):
        fraction += min(fraction_per_meter_ground_truth.values[meter_instance],
                        fraction_per_meter_predictions.values[meter_instance])

    return fraction
    
def generate_mains_power_series(mains):
    load_kwargs={}
    load_kwargs.setdefault('resample', True)
    load_kwargs.setdefault('sample_period', 60)
    load_kwargs['sections'] = mains.good_sections()
    
    chunks = list(mains.power_series(**load_kwargs))
    return chunks
    
def generate_mains_power_series_from_apps(gt_power_series, timestamps_list, vampire_power):
    sum_of_apps_power = []
    for timestamp in timestamps_list:
        sum_of_each_app_values = 0
        for app in gt_power_series.keys():
            sum_of_each_app_values += gt_power_series[app][timestamp]
        sum_of_apps_power.append(sum_of_each_app_values + vampire_power)
    
    mains_created = Series(sum_of_apps_power, index=timestamps_list)
    return mains_created
    
def generate_apps_power_series(loc):
    load_kwargs={}
    load_kwargs.setdefault('resample', True)
    load_kwargs.setdefault('sample_period', 60)
    ps = {}
    for i in  loc.min_power_threshold:
        ps[i] = list(loc.elec[i].power_series(**load_kwargs))[0]
        
    pst = dict(ps)
    if loc.name == 'REDD':
        rr1 = [sum(x) for x in zip(pst[3], pst[4])]
        rr2 = [sum(x) for x in zip(pst[10], pst[20])]
        srr1 = Series(rr1, index=ps[3].index)
        srr2 = Series(rr2, index=ps[10].index)
        pst[(3,4)] = srr1
        pst[(10,20)] = srr2
        del pst[3]
        del pst[4]
        del pst[10]
        del pst[20]
        
    return ps, pst
    
def generate_state_combinations_all(co, mains):
    from sklearn.utils.extmath import cartesian
    centroids = [model['states'] for model in co.model]
    state_combinations = cartesian(centroids)
    
    correction = 0#28.35
    vampire_power = mains.vampire_power() - correction
    print("vampire_power = {} watts".format(vampire_power))
    n_rows = state_combinations.shape[0]
    vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
    state_combinations = np.hstack((state_combinations, vampire_power_array))
    summed_power_of_each_combination = np.sum(state_combinations, axis=1)
    
    ii = [5,6,7,8,9,11,12,13,14,15,16,17,18,19,(3,4),(10,20)]
    cc = Series(centroids, index=ii)
    return vampire_power, cc, state_combinations, summed_power_of_each_combination

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    
def get_gt_state_combinations(gt_apps, loc, vampire_power, timestamp, gt_power_series, co):
    
    values = {}
    for app in gt_apps:
        values[app] = gt_power_series[app][timestamp] 
    
    try:
        v34 = values[3] + values[4]
        del values[3]
        del values[4]
        values[(3,4)] = v34
    except Exception:
        tpt = 0  
    
    try:
        v1020 = values[10] + values[20]
        del values[10]
        del values[20]
        values[(10,20)] = v1020
    except Exception:
        tpt = 0
    #gt_apps = list(gt_apps_orig)
    #Take care of REDDs tuples names (3,4) and (10,20)   
    
    
    if loc.name == 'REDD':
        if 10 in gt_apps:
            gt_apps.remove(10)
            gt_apps.remove(20)
            gt_apps.append((10,20))
        if 3 in gt_apps:
            gt_apps.remove(3)
            gt_apps.remove(4)
            gt_apps.append((3,4))
       
    centroids_gt = []
    ordering = []

    for model in co.model:
        try:
            if  model['training_metadata'].instance() in gt_apps:
                centroids_gt.append(model['states'])
                ordering.append(model['training_metadata'].instance())
        except Exception:
            for app in gt_apps:
                try:
                    if model['training_metadata'].instance() == app:
                        centroids_gt.append(model['states'])
                        ordering.append(model['training_metadata'].instance())
                except Exception:
                    continue                    
    
    #We know all these appliances are ON, take away the states when they are off
    centroids_on = {}
    for i,centroid_array in enumerate(centroids_gt):
        cd = [centroid for centroid in centroid_array if centroid != 0]
        centroids_on[gt_apps[i]] = np.array(cd)
        
    state_combinations =  [(v, find_nearest(centroids_on[v], values[v])) for v in values]   
    values_of_combination = [find_nearest(centroids_on[v], values[v]) for v in values] 
    summed_power_of_combination = sum(values_of_combination) + vampire_power

    return state_combinations, summed_power_of_combination, ordering

def get_difference(mains_from_apps,gt_sums):
    c = []
    for i, gs in enumerate(gt_sums):
        difference = float(gs) - mains_from_apps.values[i]
        c.append(difference)
    
    return c
    
def get_gt_values(chunks, loc, vampire_power, gt_power_series, co):
    locations_lists  = []
    appliances_lists = []
    timestamps_list = []
    mains_values = []
    gt = []
    gt_sums = []
    gt_residuals = []
    gt_states = []
    for chunk in chunks:
        for ts, value in enumerate(chunk):
            timestamp = chunk.index[ts]
            concurrent_events = loc.events_locations['Locations'][(timestamp - DateOffset(seconds = 60)):(timestamp)]
            concurrent_appliances = loc.events_locations['Events'][(timestamp - DateOffset(seconds = 60)):(timestamp)]
            
            gt_appliances = None
            gt_apps = []
            for gt_event_ts in loc.appliances_status.index:
                if gt_event_ts <= timestamp:
                    gt_appliances = loc.appliances_status[str(gt_event_ts)]
                    gt_ts = gt_event_ts
            if gt_appliances is not None:
                gt_apps = [v for i,v in enumerate(gt_appliances) if gt_appliances.values[0][i] == True]  
            
            
            if (len(gt_apps) == 0):
                gt.append([])
                gt_sums.append(0)
                gt_residuals.append(0)
                gt_states.append([])
            else:
                gt_state_combinations, summ, order_of_appliances = get_gt_state_combinations(
                                                                            gt_apps,
                                                                            loc, 
                                                                            vampire_power,
                                                                            timestamp,
                                                                            gt_power_series, 
                                                                            co)
                
                gt_apps1 = [v[0] for v in gt_state_combinations if v[1] not in (0,vampire_power)]
    
                gt.append(gt_apps1)
                gt_sums.append("{0:.2f}".format(summ))
                gt_residuals.append("{0:.2f}".format((summ-value)))
                 
                gt_sc = [int(v[1]) for v in gt_state_combinations if v[1] not in (0,vampire_power)]
                gt_states.append(gt_sc)
            
            
            locs = []
            [locs.extend(j) for j in concurrent_events.values]
            locations_within_timespan = list(set(locs))
            apps = []
            [apps.extend(j) for j in concurrent_appliances.values]
            appliances_within_timespan = list(set(apps))
            
            timestamps_list.append(timestamp)
            mains_values.append("{0:.2f}".format(value))
            locations_lists.append(locations_within_timespan)
            appliances_lists.append(appliances_within_timespan)
    
    gt[0] = gt[1]
    gt_states[0] = gt_states[1]        
    return locations_lists, appliances_lists, timestamps_list, mains_values, gt, gt_sums, gt_residuals, gt_states
    
def get_summed_power_of_combos(co, summed_power_of_each_combination):
    co_original_combo_sums = [ "{0:.2f}".format(summed_power_of_each_combination[index]) for index in co.co_indices_original]
    co_location_combo_sums_t = np.sum(co.co_combos_location, axis=1)
    co_location_combo_sums = [ "{0:.2f}".format(summm) for summm in co_location_combo_sums_t ]
    
    return co_original_combo_sums, co_location_combo_sums
    
def get_residuals_of_combos(co):
    co_residuals_original = ["{0:.2f}".format(residual) for residual in co.co_residuals_original]
    co_residuals_location = ["{0:.2f}".format(residual) for residual in co.co_residuals_location]
    return co_residuals_original, co_residuals_location
    
def create_sum_residual_tuples(gt_sums, gt_residuals, co_original_combo_sums, co_residuals_original, co_location_combo_sums, co_residuals_location):
    gt_sum_res  = [(gt_sums[i],gt_residuals[i])for i,v in enumerate(gt_sums)]
    co_sum_res  = [(co_original_combo_sums[i],co_residuals_original[i])for i,v in enumerate(co_original_combo_sums)]
    loc_sum_res = [(co_location_combo_sums[i],co_residuals_location[i])for i,v in enumerate(co_location_combo_sums)]
    
    return gt_sum_res, co_sum_res, loc_sum_res
    
def get_appliances_in_combos(co, loc, state_combinations):
    appliances_order = [model['training_metadata'].instance() for i, model in enumerate(co.model)]
    co_original_combos = [co.get_appliances_in_state_combination(appliances_order, state_combinations[index], loc) for index             in co.co_indices_original]
    co_location_combos = [co.get_appliances_in_state_combination(appliances_order, state_combination        , loc) for state_combination in co.co_combos_location]
    
    for combo in co_original_combos:    
        if 10 in combo:
            combo.remove(10)
            combo.remove(20)
            combo.append((10,20))
        if 3 in combo:
            combo.remove(3)
            combo.remove(4)
            combo.append((3,4))
    
    for combo in co_location_combos:    
        if 10 in combo:
            combo.remove(10)
            combo.remove(20)
            combo.append((10,20))
        if 3 in combo:
            combo.remove(3)
            combo.remove(4)
            combo.append((3,4))

    return co_original_combos, co_location_combos
    
def get_states_of_appliances_in_combos(co, vampire_power, state_combinations):
    combo_values_considered_original = [[int(v) for v in state_combinations[index] if v not in (0,vampire_power)] for index in co.co_indices_original]
    combo_values_considered_loc = [[int(v) for v in combo if v not in (0,vampire_power)] for combo in co.co_combos_location]
    return combo_values_considered_original, combo_values_considered_loc
    
def create_app_state_tuples(gt, gt_states, co_original_combos, combo_original_states, co_location_combos, combo_location_states):
    gt_combo_states = [[(gt[i_combo][i_app], gt_states[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(gt)]
    co_combo_states  = [[(co_original_combos[i_combo][i_app], combo_original_states[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(co_original_combos)]
    loc_combo_states = [[(co_location_combos[i_combo][i_app],      combo_location_states[i_combo][i_app]) for i_app, app in enumerate(combo)]for i_combo, combo in enumerate(co_location_combos)]
    return gt_combo_states, co_combo_states, loc_combo_states

def jaccard(gt_combo_states, co_combo_states, loc_combo_states):
    jaccard_co  = []
    jaccard_loc = []
    jaccard_co_states  = []
    jaccard_loc_states = []
    for i in range (0, len(gt_combo_states)):
        gt_apps =  [app_state[0] for app_state in gt_combo_states[i]]
        co_apps =  [app_state[0] for app_state in co_combo_states[i]]
        loc_apps = [app_state[0] for app_state in loc_combo_states[i]]
        
        gt_u_co  = list(set(gt_apps) | set(co_apps))
        gt_u_loc = list(set(gt_apps) | set(loc_apps))
        gt_n_co  = list(set(gt_apps) & set(co_apps))
        gt_n_loc = list(set(gt_apps) & set(loc_apps))
        
        jaccard_co.append((len(gt_n_co), len(gt_u_co)))
        jaccard_loc.append((len(gt_n_loc), len(gt_u_loc)))
            
        gt_n_co_gt_states = []
        for app in gt_n_co:
            for app_state in gt_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_co_gt_states.append(app_state[1])
                except Exception:
                    continue
              
        gt_n_co_co_states = []
        for app in gt_n_co:
            for app_state in co_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_co_co_states.append(app_state[1])
                except Exception:
                    continue
                    
        gt_n_loc_gt_states = []
        for app in gt_n_loc:
            for app_state in gt_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_loc_gt_states.append(app_state[1])
                except Exception:
                    continue
        gt_n_loc_loc_states = []
        for app in gt_n_loc:
            for app_state in loc_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_loc_loc_states.append(app_state[1])
                except Exception:
                    continue
    
        gt_n_co_states  = list(set(gt_n_co_gt_states) & set(gt_n_co_co_states))
        gt_n_loc_states = list(set(gt_n_loc_gt_states) & set(gt_n_loc_loc_states))
        
        jaccard_co_states.append((len(gt_n_co_states), len(gt_n_co)))
        jaccard_loc_states.append((len(gt_n_loc_states), len(gt_n_loc)))
    return jaccard_co, jaccard_loc, jaccard_co_states, jaccard_loc_states

def jaccard2(gt_combo_states, co_combo_states, loc_combo_states):
    jaccard_co  = []
    jaccard_loc = []
    jaccard_co_states  = []
    jaccard_loc_states = []
    for i in range (0, len(gt_combo_states)):
        gt_apps =  [app_state[0] for app_state in gt_combo_states[i]  if app_state[0] not in [7]]
        co_apps =  [app_state[0] for app_state in co_combo_states[i]  if app_state[0] not in [7]]
        loc_apps = [app_state[0] for app_state in loc_combo_states[i] if app_state[0] not in [7]]
        
        gt_u_co  = list(set(gt_apps) | set(co_apps))
        gt_u_loc = list(set(gt_apps) | set(loc_apps))
        gt_n_co  = list(set(gt_apps) & set(co_apps))
        gt_n_loc = list(set(gt_apps) & set(loc_apps))
        
        jaccard_co.append((len(gt_n_co), len(gt_u_co)))
        jaccard_loc.append((len(gt_n_loc), len(gt_u_loc)))
            
        gt_n_co_gt_states = []
        for app in gt_n_co:
            for app_state in gt_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_co_gt_states.append(app_state[1])
                except Exception:
                    continue
              
        gt_n_co_co_states = []
        for app in gt_n_co:
            for app_state in co_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_co_co_states.append(app_state[1])
                except Exception:
                    continue
                    
        gt_n_loc_gt_states = []
        for app in gt_n_loc:
            for app_state in gt_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_loc_gt_states.append(app_state[1])
                except Exception:
                    continue
        gt_n_loc_loc_states = []
        for app in gt_n_loc:
            for app_state in loc_combo_states[i]:
                try:
                    if app_state[0] == app:
                        gt_n_loc_loc_states.append(app_state[1])
                except Exception:
                    continue
    
        gt_n_co_states  = list(set(gt_n_co_gt_states) & set(gt_n_co_co_states))
        gt_n_loc_states = list(set(gt_n_loc_gt_states) & set(gt_n_loc_loc_states))
        
        jaccard_co_states.append((len(gt_n_co_states), len(gt_n_co)))
        jaccard_loc_states.append((len(gt_n_loc_states), len(gt_n_loc)))
    return jaccard_co, jaccard_loc, jaccard_co_states, jaccard_loc_states
    
def jaccard_total(jaccard_co, jaccard_loc, jaccard_co_states, jaccard_loc_states):
    n_jacc_app_co = 0
    n_jacc_app_loc = 0
    n_jacc_states_co = 0
    n_jacc_states_loc = 0
    
    n_jacc_app_gt_co = 0
    n_jacc_app_gt_loc = 0
    n_jacc_states_gt_co = 0
    n_jacc_states_gt_loc = 0
    
    accum_jacc_apps_co    = 0
    accum_jacc_apps_loc   = 0
    accum_jacc_states_co  = 0
    accum_jacc_states_loc = 0
    accum_jacc_statesapp_co  = 0
    accum_jacc_statesapp_loc = 0
    for i in range(0, len(jaccard_co)):
        jacc_app_co = jaccard_co[i][0]
        jacc_app_gt_co = jaccard_co[i][1]
        
        jacc_app_loc = jaccard_loc[i][0]
        jacc_app_gt_loc = jaccard_loc[i][1]
        
        jacc_states_co = jaccard_co_states[i][0]
        jacc_states_gt_co = jaccard_co_states[i][1]
        
        jacc_states_loc = jaccard_loc_states[i][0] 
        jacc_states_gt_loc = jaccard_loc_states[i][1] 
        
        accum_jacc_apps_co  += float(jacc_app_co)/float(jacc_app_gt_co)
        accum_jacc_apps_loc += float(jacc_app_loc)/float(jacc_app_gt_loc)
        
        accum_jacc_states_co  += float(jacc_states_co)/float(jacc_states_gt_co)
        accum_jacc_states_loc += float(jacc_states_loc)/float(jacc_states_gt_loc)        
        
        accum_jacc_statesapp_co  += float(jacc_states_co)/float(jacc_app_gt_co)
        accum_jacc_statesapp_loc += float(jacc_states_loc)/float(jacc_app_gt_loc)
        
        n_jacc_app_co        += jacc_app_co
        n_jacc_app_loc       += jacc_app_loc
        n_jacc_states_co     += jacc_states_co
        n_jacc_states_loc    += jacc_states_loc
        n_jacc_app_gt_co     += jacc_app_gt_co
        n_jacc_app_gt_loc    += jacc_app_gt_loc
        n_jacc_states_gt_co  += jacc_states_gt_co
        n_jacc_states_gt_loc += jacc_states_gt_loc
    

    avg_jacc_apps_co    = float(accum_jacc_apps_co)/ float(len(jaccard_co))     
    avg_jacc_apps_loc   = float(accum_jacc_apps_loc)/ float(len(jaccard_co))     
    avg_jacc_states_co  = float(accum_jacc_states_co)/ float(len(jaccard_co))     
    avg_jacc_states_loc = float(accum_jacc_states_loc)/ float(len(jaccard_co))
    #print accum_jacc_apps_co, accum_jacc_apps_loc, accum_jacc_states_co, accum_jacc_states_loc, len(jaccard_co), accum_jacc_statesapp_co, accum_jacc_statesapp_loc
    
    ptg_co_apps = "{0:.2f}%".format(100* avg_jacc_apps_co)
    ptg_loc_apps = "{0:.2f}%".format(100* avg_jacc_apps_loc)
    ptg_co_states = "{0:.2f}%".format(100* avg_jacc_states_co)
    ptg_loc_states = "{0:.2f}%".format(100* avg_jacc_states_loc)
    
#    ptg_co_apps = "{0:.2f}%".format(100* n_jacc_app_co/n_jacc_app_gt_co)
#    ptg_loc_apps = "{0:.2f}%".format(100* n_jacc_app_loc/n_jacc_app_gt_loc)
#    ptg_co_states = "{0:.2f}%".format(100* n_jacc_states_co/n_jacc_states_gt_co)
#    ptg_loc_states = "{0:.2f}%".format(100* n_jacc_states_loc/n_jacc_states_gt_loc)
    jaccard_results = {
                        'CO apps':((n_jacc_app_co,n_jacc_app_gt_co),ptg_co_apps),
                        'Loc apps':((n_jacc_app_loc,n_jacc_app_gt_loc),ptg_loc_apps),
                        'CO states':((n_jacc_states_co,n_jacc_states_gt_co),ptg_co_states),
                        'Loc states':((n_jacc_states_loc,n_jacc_states_gt_loc),ptg_loc_states)
                       }

    return jaccard_results  
        
def create_jaccard_apps_states_tuples(jaccard_co, jaccard_loc, jaccard_co_states, jaccard_loc_states):
    jaccard_co_apps_states = []
    jaccard_loc_apps_states = []
    for i in range(0, len(jaccard_co)):    
        jaccard_co_apps_states.append((jaccard_co[i],jaccard_co_states[i]))
        jaccard_loc_apps_states.append((jaccard_loc[i],jaccard_loc_states[i]))
    return jaccard_co_apps_states, jaccard_loc_apps_states
    
def build_results_table(locations_lists, appliances_lists, 
                        gt_combo_states, co_combo_states, loc_combo_states,
                        mains_values, gt_sum_res, co_sum_res, loc_sum_res,
                        jacc_co_apps_states, jacc_loc_apps_states,
                        timestamps_list):
    comparison = {}
    comparison['01. Locs w/event'] = locations_lists
    comparison['02. Apps w/event'] = appliances_lists
    comparison['03. GT combo/states'] = gt_combo_states
    comparison['04. CO combo/states'] = co_combo_states
    comparison['05. Loc combo/states'] = loc_combo_states
    comparison['06. Mains'] = mains_values
    comparison['07. GT sum/res'] = gt_sum_res
    comparison['08. CO sum/res'] = co_sum_res
    comparison['09. Loc sum/res'] = loc_sum_res
    comparison['10. CO jacc app/states'] = jacc_co_apps_states
    comparison['11. Loc jacc app/states'] = jacc_loc_apps_states
        
    d = DataFrame(comparison, index=timestamps_list)
    return d

def build_metrics_tables(fraction_original, fraction_loc, jaccard_results, total_error_co, total_error_loc,
                         proportion_error_co, proportion_error_loc, normal_error_co, normal_error_loc,
                         gt_proportion_co, pr_proportion_co, pr_proportion_loc):
    data = {'co' : [fraction_original, jaccard_results['CO apps'], jaccard_results['CO states'],  total_error_co], 'loc': [fraction_loc, jaccard_results['Loc apps'], jaccard_results['Loc states'], total_error_loc]}
    m  = DataFrame(data, index=['Fraction', 'Jaccard Apps', 'Jaccard States', 'Total error'])
    
    dnepe_co_loc = {'01. co PE' : proportion_error_co, '02. loc PE': proportion_error_loc, '03. co NE' : normal_error_co, '04. loc NE': normal_error_loc}
    maa  = DataFrame(dnepe_co_loc) 
    
    dpe_co_loc  = {'01. % Tru' : gt_proportion_co, '02. % co': pr_proportion_co, '03. % loc': pr_proportion_loc, '04. PE co': proportion_error_co, '05. PE loc': proportion_error_loc}
    ma   = DataFrame(dpe_co_loc) 
    ma['06. NE co'] = maa['03. co NE']
    ma['07. NE loc'] = maa['04. loc NE']
    return m, ma
    
def compare_mains_and_gt_of_appliances(gt_power_series, mains_values, timestamps_list, vampire_power, gt_pst):
    sum_of_apps_power = []
    for timestamp in timestamps_list:
        sum_of_each_app_values = 0
        for app in gt_power_series.keys():
            sum_of_each_app_values += gt_power_series[app][timestamp]
        sum_of_apps_power.append(sum_of_each_app_values + vampire_power)

    comparison_mains_and_apps_abs = []
    comparison_mains_and_apps = []
    for i,summ in enumerate(sum_of_apps_power):
        diff = sum_of_apps_power[i] - float(mains_values[i])
        comparison_mains_and_apps_abs.append(abs(diff))
        comparison_mains_and_apps.append(diff)
        
    fmains = [float(value) for value in mains_values]
    t = {}
    t['summ of apps'] = sum_of_apps_power
    t['mains'] = fmains
    #t['diff'] = comparison_mains_and_apps

    tt = {}
    tt['diff'] = comparison_mains_and_apps
    tt['diffabs'] = comparison_mains_and_apps_abs
    
    mains_from_apps = Series(sum_of_apps_power, index=timestamps_list)
    smains = Series(fmains, index=timestamps_list)
    #diffs       = Series(comparison_mains_and_apps, index=timestamps_list)
    
    diffs = DataFrame(tt, index=timestamps_list)    
    d = DataFrame(t, index=timestamps_list)
    
    gt_all = DataFrame(gt_pst)
    gt_all["sum"] = gt_all.sum(axis=1)
    gt_all["mains_apps"] = gt_all['sum'] + vampire_power
    gt_all['mains'] = smains
    gt_all["diff1"] = abs(gt_all['mains_apps'] - gt_all['mains'])
    gt_all["diff2"] = abs(gt_all['sum'] - gt_all['mains'])

    return d, mains_from_apps, smains, diffs, gt_all
    

def get_apps(loc):
    apps = loc.appliances_location.keys()
    if loc.name == 'REDD':
        apps.remove(3)
        apps.remove(4)
        apps.remove(10)
        apps.remove(20)
        apps.append((3,4))
        apps.append((10,20))
    return apps
    
def get_predicted_values_from_combos_found(loc, combo_states):
    predicted_values = {}
    apps = get_apps(loc)
        
    for app in apps:
        predicted_values[app] = []
        
    for instance in combo_states:
        apps_in_combo = [tup[0] for tup in instance]
        for app in apps:
            if app in apps_in_combo:
                for tup in instance:
                    if(tup[0] == app):
                        v = tup[1]
            else:
                v = 0
            predicted_values[app].append(v)
   
    return predicted_values
       
def proportion_error_per_appliance(loc, mains_values, gt_values, predicted_values):
    gt_proportion = {}
    pr_proportion = {}
    proportion_error = {}
    
    apps = get_apps(loc)
    for app in apps:
        proportion_gt = []
        proportion_pr = []
        for i,value in enumerate(mains_values):
            ts = predicted_values[app].index[i]
            p_gt = gt_values[app][ts]/float(mains_values[i])
            p_pr = predicted_values[app][ts]/float(mains_values[i])
            proportion_gt.append(p_gt)
            proportion_pr.append(p_pr)
                    
        summ_gt = sum(proportion_gt)
        summ_pr = sum(proportion_pr)

        T = len(mains_values)
        tru = summ_gt/T
        dis = summ_pr/T
                
        gt_proportion[app] = tru
        pr_proportion[app] = dis
        
        diff = abs(tru - dis)
        proportion_error[app] = diff
    return proportion_error, gt_proportion, pr_proportion

def proportion_error_per_appliance_df(mains_values, gt_values, predicted_values):
    gt_proportion = {}
    pr_proportion = {}
    proportion_error = {}

    for app in predicted_values:
        p_gt  = gt_values[app]/mains_values
        p_pr  = predicted_values[app]/mains_values
        
        fr = DataFrame(p_gt, columns=['p_gt'])
        fr['p_gt'] = p_gt
        fr['p_pr'] = p_pr
        
#        fr['01. mains'] = mains_values
#        fr['02. gt'] = gt_values[app]
#        fr['03. pr'] = predicted_values[app]
        
        fr = fr.dropna()
        
        summ_gt = fr['p_gt'].sum()
        summ_pr = fr['p_pr'].sum()

        T = len(fr)
        tru = float(summ_gt)/float(T)
        dis = float(summ_pr)/float(T)
                
        gt_proportion[app] = tru
        pr_proportion[app] = dis
        
        diff = abs(tru - dis)
        proportion_error[app] = diff
    return proportion_error, gt_proportion, pr_proportion

def normal_disaggregation_error_per_appliance(loc, gt_values, predicted_values):
    #NOT OK
    normal_error = {}
    squares_of_diffs = {}
    squares_of_gts = {}
    
    apps = get_apps(loc)
    for app in apps:
        square_of_difference = []
        square_of_gt = []
        for i,value in enumerate(predicted_values[app]):
            ts = predicted_values[app].index[i]
            diff = gt_values[app][ts] - predicted_values[app][ts]
            sq_diff = pow(diff,2)
            sq_gt   = pow(value,2)
            square_of_difference.append(sq_diff)
            square_of_gt.append(sq_gt)
                           
        summ_sq_diff = sum(square_of_difference)
        summ_sq_gt   = sum(square_of_gt)

        squares_of_diffs[app] = summ_sq_diff
        squares_of_gts[app]   = summ_sq_gt
        
        nerror_sq = summ_sq_diff/summ_sq_gt
        
        nerror = sqrt(nerror_sq)
        normal_error[app] = nerror
    return normal_error, squares_of_diffs, squares_of_gts
    
def normal_disaggregation_error_per_appliance_df(gt_values, predicted_values):
    normal_error = {}
    sqrs = {}
    for app in predicted_values:
        dd  = gt_values[app] - predicted_values[app]
        dd2 = dd**2        
        g2  = gt_values[app]**2
        
        fr = DataFrame(dd2, columns=['num'])
        fr['num'] = dd2
        fr['den'] = g2
        fr = fr.dropna()

        nerror_sq = fr['num'].sum()/fr['den'].sum()
        nerror = sqrt(nerror_sq)
        normal_error[app] = nerror
        
        sqrs[app] = fr
    return normal_error, sqrs 

def total_disaggregation_error(loc, squares_of_diffs, squares_of_gts):
    #NOT OK
    total_error = 0
    apps = get_apps(loc)
    
    sum_sq_of_diffs = 0
    sum_sq_of_gts = 0
    for app in apps:
        sum_sq_of_diffs += squares_of_diffs[app]
        sum_sq_of_gts   += squares_of_gts[app]
    
    t_error_sq =  sum_sq_of_diffs/sum_sq_of_gts
    total_error = sqrt(t_error_sq)
    return total_error    
    
def total_disaggregation_error_df(gt_values, predicted_values):
    total_error = 0
    num_sum = 0
    den_sum = 0
    for app in predicted_values:
        dd  = gt_values[app] - predicted_values[app]
        dd2 = dd**2        
        g2  = gt_values[app]**2
        
        fr = DataFrame(dd2, columns=['num'])
        fr['num'] = dd2
        fr['den'] = g2
        fr = fr.dropna()

        num_sum += fr['num'].sum()
        den_sum += fr['den'].sum()
    
    t_error_sq = num_sum/den_sum    
    total_error = sqrt(t_error_sq)
    return total_error    

def predicted_values_to_series(t, timestamps_list):  
    series = {}
    for i, app in enumerate(t):
        d = Series(t[app], index=timestamps_list) 
        series[app] = d
    return series    

def save_to_files(fnpath, d, m, ma):
    d.to_csv(fnpath+'_eval.csv')
    m.to_csv(fnpath+'_metrics.csv')
    ma.to_csv(fnpath+'_metrics_per_appliance.csv')
    
    
def save_objects(fnpath, d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs):
    objects = {}
    
    objects['eval'] = d
    objects['metrics'] = m
    objects['metrics app'] = ma
    
    objects['gt'] = gt_pst
    objects['dis_co'] = dis_co
    objects['dis_loc'] = dis_loc
    objects['mains'] = smains
    objects['mains_apps'] = mains_from_apps
    objects['mains_diffs'] = diffs
    
        
    pickle.dump( objects, open(fnpath+'_dump.p', "wb" ) )
    
def read_objects(fnpath):
    objects = pickle.load(open(fnpath+'_dump.p', "rb"))
    return objects
    
def dismantle_object(objects):
    d   = objects['eval']
    m   = objects['metrics']
    ma  = objects['metrics app']
    
    gt_pst  = objects['gt']
    dis_co  = objects['dis_co']
    dis_loc = objects['dis_loc']
    smains  = objects['mains']
    mains_from_apps = objects['mains_apps']
    diffs   = objects['mains_diffs']
    
    return d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs