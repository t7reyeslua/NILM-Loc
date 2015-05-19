# -*- coding: utf-8 -*-

from math import sqrt
import nilmtk.metrics as nilmtk_metrics
from nilmtk.metergroup import MeterGroup
from pandas import DataFrame, Series
import numpy as np 
import cPickle as pickle

def f1_score(disag, original):
    f1_score = nilmtk_metrics.f1_score(disag, original)
    return f1_score

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
        ptg_co_apps  = "{0:.2f}".format(float(jaccard_co[i][0]) /float(jaccard_co[i][1]))
        ptg_loc_apps = "{0:.2f}".format(float(jaccard_loc[i][0])/float(jaccard_loc[i][1]))
        
        ptg_co_apps_states = "{0:.2f}".format(float(jaccard_co_states[i][0])/float(jaccard_co_states[i][1]))
        ptg_loc_apps_states = "{0:.2f}".format(float(jaccard_loc_states[i][0])/float(jaccard_loc_states[i][1]))
        
        jacc_res_co  = [jaccard_co[i], ptg_co_apps, jaccard_co_states[i], ptg_co_apps_states]
        jacc_res_loc = [jaccard_loc[i], ptg_loc_apps, jaccard_loc_states[i], ptg_loc_apps_states]
       
#        jaccard_co_apps_states.append((jaccard_co[i],jaccard_co_states[i]))
#        jaccard_loc_apps_states.append((jaccard_loc[i],jaccard_loc_states[i]))
        
        jaccard_co_apps_states.append(jacc_res_co)
        jaccard_loc_apps_states.append(jacc_res_loc)
        
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

    
def get_predicted_values_from_combos_found(loc, combo_states):
    predicted_values = {}
    apps = loc.metadata.get_apps()
        
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
    
    apps = loc.metadata.get_apps()
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
    
    apps = loc.metadata.get_apps()
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
    apps = loc.metadata.get_apps()
    
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
    
class Metrics(object):

    def __init__(self, co, ground_truth, loc, disag_elec_location, disag_elec_original):
        self.co  = co
        self.gt  = ground_truth
        self.loc = loc
        self.metergroup = loc.elec
        self.disag_elec_location = disag_elec_location        
        self.disag_elec_original = disag_elec_original        
        
        self.f1_original = None
        self.f1_location = None
        self.f1_difference_original_and_location = None  
        
        self.fraction_assigned_correctly_original = None
        self.fraction_assigned_correctly_location = None
        
        self.gt_sum_res = None
        self.co_sum_res = None
        self.loc_sum_res = None
        self.co_original_combos = None
        self.co_location_combos = None
        self.co_original_states = None
        self.co_location_states = None
        
        self.gt_combo_states = None
        self.co_combo_states = None
        self.loc_combo_states = None
        
        self.jacc_co = None
        self.jacc_loc = None
        self.jacc_co_states = None
        self.jacc_loc_states = None
        self.jacc_co_apps_states = None
        self.jacc_loc_apps_states = None
        self.jaccard_results = None
                
        self.power_series_apps_table_co = None       
        self.power_series_apps_table_loc = None 
        
        self.proportion_error_co = None
        self.gt_proportion_co = None
        self.pr_proportion_co = None
        self.proportion_error_loc = None
        self.gt_proportion_loc = None
        self.pr_proportion_loc = None
        
        self.normal_error_co = None
        self.sqrs_co = None
        self.normal_error_loc = None
        self.sqrs_loc = None
        self.total_error_co = None
        self.total_error_loc = None
        
        self.results_disaggregation = None
        self.results_metrics = None   
        self.results_metrics_appliances = None  
        return
        
    def calculate_f1_score(self):
        self.f1_original = f1_score(self.disag_elec_original, self.metergroup)
        self.f1_location = f1_score(self.disag_elec_location, self.metergroup)
        
        diff1  = [self.f1_location.values[i] - self.f1_original.values[i] for i,v in enumerate(self.f1_location)]
        self.f1_difference_original_and_location = Series(diff1, index=self.f1_location.index)             
        return
        
    def calculate_fraction_energy_assigned_correctly(self):
        self.fraction_assigned_correctly_original = fraction_energy_assigned_correctly(self.disag_elec_original, self.metergroup)
        self.fraction_assigned_correctly_location = fraction_energy_assigned_correctly(self.disag_elec_location, self.metergroup)
        return
        
    def calculate_sum_and_residual_of_found_combos(self):
        #Sums of found combos and corresponding residuals
        self.co_combo_sums_original, self.co_combo_sums_location            = get_summed_power_of_combos(self.co, self.gt.summed_power_of_each_combination)
        self.co_combo_residuals_original, self.co_combo_residuals_location  = get_residuals_of_combos(self.co)
        self.gt_sum_res, self.co_sum_res, self.loc_sum_res                = create_sum_residual_tuples(self.gt.gt_appliances_summed_power, 
                                                                                        self.gt.gt_appliances_residual, 
                                                                                        self.co_combo_sums_original, 
                                                                                        self.co_combo_residuals_original, 
                                                                                        self.co_combo_sums_location, 
                                                                                        self.co_combo_residuals_location)
        return

    def get_app_and_states_of_combos(self):
        #Appliances and states guessed
        self.co_original_combos, self.co_location_combos   = get_appliances_in_combos(self.co, self.loc, self.gt.state_combinations)
        self.co_original_states, self.co_location_states   = get_states_of_appliances_in_combos(self.co, self.gt.vampire_power, self.gt.state_combinations)
        self.gt_combo_states, self.co_combo_states, self.loc_combo_states = create_app_state_tuples(self.gt.gt_appliances, 
                                                                                     self.gt.gt_appliances_states, 
                                                                                     self.co_original_combos, 
                                                                                     self.co_original_states, 
                                                                                     self.co_location_combos, 
                                                                                     self.co_location_states)
        return
                                                                                     
                                                                                
    def calculate_jaccard(self):     

        if not all([self.gt_combo_states, self.co_combo_states, self.loc_combo_states]):
            print ('Run first: get_app_and_states_of_combos')
            return
            
        #Jaccard
        self.jacc_co, self.jacc_loc, self.jacc_co_states, self.jacc_loc_states = jaccard(self.gt_combo_states, self.co_combo_states, self.loc_combo_states)
        self.jacc_co_apps_states, self.jacc_loc_apps_states     = create_jaccard_apps_states_tuples(self.jacc_co, self.jacc_loc, self.jacc_co_states, self.jacc_loc_states)
        self.jaccard_results                                    = jaccard_total(self.jacc_co, self.jacc_loc, self.jacc_co_states, self.jacc_loc_states)
        
        return
        
    def get_apps_power_series_from_combos_found(self):
        pr_co = get_predicted_values_from_combos_found(self.loc, self.co_combo_states)
        pr_loc = get_predicted_values_from_combos_found(self.loc, self.loc_combo_states)
        
        spr_co  = predicted_values_to_series(pr_co, self.gt.timestamps)
        spr_loc = predicted_values_to_series(pr_loc, self.gt.timestamps)
        dis_co = DataFrame(spr_co)
        dis_loc = DataFrame(spr_loc)
                
        self.power_series_apps_table_co = dis_co       
        self.power_series_apps_table_loc = dis_loc       
        
        return
        
    def calculate_proportion_error(self):        
        #Proportion Error per appliance
        self.proportion_error_co,  self.gt_proportion_co,  self.pr_proportion_co   = proportion_error_per_appliance_df(self.gt.power_series_mains_with_timestamp, self.gt.power_series_apps_table, self.power_series_apps_table_co)
        self.proportion_error_loc, self.gt_proportion_loc, self.pr_proportion_loc  = proportion_error_per_appliance_df(self.gt.power_series_mains_with_timestamp, self.gt.power_series_apps_table, self.power_series_apps_table_loc)
        return
                
    def calculate_normal_disaggregation_error(self):        
        #Normal Disaggregation Error per appliance
        self.normal_error_co, self.sqrs_co   = normal_disaggregation_error_per_appliance_df(self.gt.power_series_apps_table, self.power_series_apps_table_co)
        self.normal_error_loc, self.sqrs_loc = normal_disaggregation_error_per_appliance_df(self.gt.power_series_apps_table, self.power_series_apps_table_loc)

        return
                
    def calculate_total_disaggregation_error(self):        
        #Total Disaggregation Error
        self.total_error_co  = total_disaggregation_error_df(self.gt.power_series_apps_table, self.power_series_apps_table_co)
        self.total_error_loc = total_disaggregation_error_df(self.gt.power_series_apps_table, self.power_series_apps_table_loc)
        return
        
    def calculate(self):
        self.calculate_f1_score()
        self.calculate_fraction_energy_assigned_correctly()
        self.calculate_sum_and_residual_of_found_combos()
        self.get_app_and_states_of_combos()
        self.calculate_jaccard()
        self.get_apps_power_series_from_combos_found()
        self.calculate_proportion_error()
        self.calculate_normal_disaggregation_error()
        self.calculate_total_disaggregation_error()
        return

    def build_results_tables(self):
        #Build dataframes to show results more clearly
        d = build_results_table(self.gt.event_locations, self.gt.event_appliances, 
                                self.gt_combo_states, self.co_combo_states, self.loc_combo_states,
                                self.gt.mains_values, 
                                self.gt_sum_res, self.co_sum_res, self.loc_sum_res,
                                self.jacc_co_apps_states, self.jacc_loc_apps_states,
                                self.gt.timestamps)
        
        m, ma = build_metrics_tables(self.fraction_assigned_correctly_original, self.fraction_assigned_correctly_location, 
                                     self.jaccard_results, 
                                     self.total_error_co, self.total_error_loc, 
                                     self.proportion_error_co, self.proportion_error_loc, 
                                     self.normal_error_co, self.normal_error_loc,
                                     self.gt_proportion_co, self.pr_proportion_co, self.pr_proportion_loc)
        
        self.results_disaggregation = d
        self.results_metrics = m   
        self.results_metrics_appliances = ma                      
        return
                         
    def save_to_files(self, fn):
        save_to_files(fn, self.results_disaggregation, self.results_metrics, self.results_metrics_appliances)
        return

    def save_objects(self, fn):
        save_objects(fn,
                     self.results_disaggregation, self.results_metrics, self.results_metrics_appliances,
                     self.power_series_apps_table, self.power_series_apps_table_co, self.power_series_apps_table_loc,
                     self.gt.power_series_mains_with_timestamp, self.gt.power_series_mains_from_apps,
                     self.gt.comparison)
        return

    def read_objects(self, fn):
        objects = read_objects(fn)
        return objects
        
        
##Read objects
#r1_objects = metrics.read_objects(fn_path + fn_obj)
#d, m, ma, gt_pst, dis_co, dis_loc, smains, mains_from_apps, diffs = metrics.dismantle_object(r1_objects)