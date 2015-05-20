# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:00:01 2015

Combinatorial Optimization using location constraints. Based on the code from
NILMTK combinatorial_optimisation.py.

@author: t7
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
from pandas import DateOffset, Series
from datetime import datetime
from nilmtk.feature_detectors import cluster
from nilmtk.timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from nilmtk.utils import find_nearest
import pprint as pp
import gc

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class CombinatorialOptimisation(object):

    def __init__(self):
        self.model = []
           
    
    def train(self, metergroup, max_num_clusters = 3, resample_seconds=60, centroids=None):
        self.model = []       
        
        if centroids is None:            
            load_kwargs={}
            load_kwargs.setdefault('resample', True)
            load_kwargs.setdefault('sample_period', resample_seconds) 
            for i, meter in enumerate(metergroup.submeters().meters):
                print("Training model for submeter '{}'".format(meter))
                for chunk in meter.power_series(**load_kwargs):
                    states = cluster(chunk, max_num_clusters)
                    self.model.append({
                        'states': states,
                        'training_metadata': meter})
                    break
        else:
            for i, meter in enumerate(metergroup.submeters().meters):
                print("Setting model for submeter '{}'".format(meter))
                states = centroids[meter.instance()]
                self.model.append({
                    'states': states,
                    'training_metadata': meter})
            
        print("Done training!")
       

    def get_ordering_of_appliances_in_state_combinations_table(self):
        appliances_order = [model['training_metadata'].instance() for i, model in enumerate(self.model)]        
        return appliances_order
           

    def get_locations_of_events_within_timespan(self, location_data, timestamp, max_time_difference = 60):
        # Get all events that fall within +- the maximum_time_difference
    
        self.maximum_time_difference = max_time_difference #seconds        
        concurrent_events = location_data.locations[
        timestamp:
        (timestamp + DateOffset(seconds = self.maximum_time_difference))]# + DateOffset(seconds = self.maximum_time_difference))]
        
        locs = []
        [locs.extend(j) for j in concurrent_events.values]
        locations_within_timespan = list(set(locs))
        
        return locations_within_timespan
                  
    def get_locations_of_appliances(self, appliances, location_data, user_dependent_only = False):
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_list = []
        for app in appliances:
            if type(app) is tuple:
                appliances_list.extend([y for y in app])
            else:
                appliances_list.append(app)   
        
        locations = []
        if (user_dependent_only):
            [locations.extend(location_data.metadata.appliances_location[app]) for app in appliances_list if (app in location_data.metadata.user_dependent_appliances)]
        else:
            [locations.extend(location_data.metadata.appliances_location[app]) for app in appliances_list]
        return list(set(locations))
        
         
    def get_appliances_and_locations_in_state_combination(self, appliances_order, state_combination, location_data):
        #Remove vampire power column
        state_combination = state_combination[:-1]
        appliances_in_state_combination_temp = [appliances_order[v] for v,j in enumerate(state_combination) if ((j != 0))]   
        
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_in_state_combination = []
        for app in appliances_in_state_combination_temp:
            if type(app) is tuple:
                appliances_in_state_combination.extend([y for y in app])
            else:
                appliances_in_state_combination.append(app)                
        
        locations = []
        [locations.extend(location_data.metadata.appliances_location[app]) for app in appliances_in_state_combination]        
        return appliances_in_state_combination, list(set(locations)) 
       
        
    def get_appliances_in_state_combination(self, appliances_order, state_combination, location_data):
        #Remove vampire power column
        state_combination = state_combination[:-1]
        appliances_in_state_combination_temp = [appliances_order[v] for v,j in enumerate(state_combination) if ((j != 0))]        
        
        #Disintegrate any tuple into its individual elements if a tuple exists
        appliances_in_state_combination = []
        for app in appliances_in_state_combination_temp:
            if type(app) is tuple:
                appliances_in_state_combination.extend([y for y in app])
            else:
                appliances_in_state_combination.append(app)                
        
        return appliances_in_state_combination

    def get_full_state_combination(self, incomplete_combo, order_of_incomplete):
        appliances_order = self.get_ordering_of_appliances_in_state_combinations_table()
        state_combination = []
        for app in appliances_order:
            if app in order_of_incomplete:
                state_combination.append(incomplete_combo[order_of_incomplete.index(app)])
            else:
                state_combination.append(0)
        return np.array(state_combination)

    def get_appliances_that_changed(self, state_combination_original, last_state_combination_original, appliances_order, last_order_of_appliances):
        #Make both the same size to be able to compare them        
        state_combination = self.get_full_state_combination(state_combination_original, appliances_order)        
        last_state_combination = self.get_full_state_combination(last_state_combination_original, last_order_of_appliances)        
        appliances_order_full = self.get_ordering_of_appliances_in_state_combinations_table()
        
        print ('Full combo', str(state_combination))
        print ('Full last combo', str(last_state_combination))
        
        changed_appliances_bool_array = np.isclose(state_combination, last_state_combination)
        index_of_changed_appliances = [i for i, closeEnough in enumerate(changed_appliances_bool_array) if closeEnough == False]
        
        changed_appliances_ON  = [appliances_order_full[i] for i in index_of_changed_appliances if state_combination[i] > last_state_combination[i]]
        changed_appliances_OFF = [appliances_order_full[i] for i in index_of_changed_appliances if state_combination[i] < last_state_combination[i]]
        
        return changed_appliances_ON, changed_appliances_OFF

        
    def check_if_valid_state_combination(self, state_combination, last_state_combination, valid_locations, appliances_order, location_data, last_order_of_appliances):        
        # We cannot turn ON any appliance if its location (location_data.appliances_location)
        # is not found in the valid_locations. For determining if we are turning ON an appliance
        # we need to compare the found combination with the last_state_combination and check
        # if there is an appliance ON that was OFF the last time.
                                         
                                         
        # We cannot turn OFF any appliance if its location (location_data.appliances_location)
        # is not found in the valid_locations. For determining if we are turning OFF an appliance
        # we need to compare the found combination with the last_state_combination and check
        # if there is an appliance OFF that was ON the last time. This only applies to 
        # appliances in the list of user dependent appliances (location_data.user_dependent_appliances)
                                         
                                         
        changed_to_ON, changed_to_OFF = self.get_appliances_that_changed(state_combination, last_state_combination, appliances_order, last_order_of_appliances)
        
        locations_ON  = self.get_locations_of_appliances(changed_to_ON, location_data)
        locations_OFF_user_dependent = self.get_locations_of_appliances(changed_to_OFF, location_data, user_dependent_only=True)
        locations_of_changes = locations_ON + locations_OFF_user_dependent

        valid_state_combination = all(location in valid_locations for location in locations_of_changes)
        return valid_state_combination, locations_of_changes    

     
    def get_constrained_state_combinations(self, valid_locations, last_combination_appliances, loc, vampire_power):
        #This method constructs only the valid state combinations from the beginning.
        
        #TODO any or all
        appliances_in_valid_locations_temp = [app for app in loc.metadata.appliances_location if all(locs in loc.metadata.appliances_location[app] for locs in valid_locations)]
        appliances_in_valid_locations_temp.extend(last_combination_appliances)
        
        #Fridge mayalways start running
        #TODO append 5 
        #TODO include always consuming appliances
        appliances_in_valid_locations_temp.append(5)
        
        appliances_in_valid_locations = list(set(appliances_in_valid_locations_temp))
        
        #Take care of REDDs tuples names (3,4) and (10,20)
        if loc.name == 'REDD':
            if 10 in appliances_in_valid_locations:
                appliances_in_valid_locations.remove(10)
                appliances_in_valid_locations.remove(20)
                appliances_in_valid_locations.append((10,20))
            if 3 in appliances_in_valid_locations:
                appliances_in_valid_locations.remove(3)
                appliances_in_valid_locations.remove(4)
                appliances_in_valid_locations.append((3,4))
           
        centroids = [model['states'] for model in self.model if  model['training_metadata'].instance() in appliances_in_valid_locations]
        ordering  = [model['training_metadata'].instance() for model in self.model if  model['training_metadata'].instance() in appliances_in_valid_locations]

        from sklearn.utils.extmath import cartesian
        state_combinations = cartesian(centroids)
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)

        return state_combinations, summed_power_of_each_combination, ordering

        
    def get_tolerance_margin(self, test_value):
        tolerance_margin = 15
        if test_value > 1000:
            tolerance_margin = 100
        elif test_value > 500:
            tolerance_margin = 50
        elif test_value > 200:
            tolerance_margin = 25
        return tolerance_margin
        
             
    def get_index_of_last_combination_in_current_state_combinations(self, last_state_combination, state_combinations):
        index = -1
        if (not isinstance(last_state_combination, int)):
            if (len(state_combinations[0]) == len(last_state_combination)):
                for i,state_combination in enumerate(state_combinations):
                    if (last_state_combination==state_combination).all():
                        index = i
                        break
        return index
        
    def find_nearest_original_co(self, known_array_sorted, test_value, index_sorted, known_array, priority_index):
        idx1 = np.searchsorted(known_array_sorted, test_value)
        idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
        idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)
    
        diff1 = known_array_sorted[idx3] - test_value
        diff2 = test_value - known_array_sorted[idx2]
    
        index = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
        residual = test_value - known_array[index]
        
        priority_enabled = False
        if (priority_enabled):
            if (priority_index not in (-1, index)): #only if it is a valid index or it has not yet selected as index
                residual_priority = test_value - known_array[priority_index]
                print('Checking priority combo...')
                if abs(residual_priority) < 10:                
                    print('...using priority', residual, residual_priority)
                    index = priority_index
                    residual = residual_priority
        
        return index, residual

        
        
    def find_nearest_location_constraint(self, 
                                         summed_power_of_each_combination, 
                                         test_value, 
                                         location_data, 
                                         state_combinations, 
                                         last_state_combination, 
                                         appliances_order,
                                         valid_locations,
                                         vampire_power,
                                         last_order_of_appliances):
        
        tolerance_margin = self.get_tolerance_margin(test_value)
        valid_combination = False
        first_guess = -1
        
                    
        #Find the last_combo in the current state combinations
        #This combo should have "priority", i.e. we should stick with that combo if nothing has changed
        #instead of just keep moving between combos with similar summed_power
        priority_combo = self.get_index_of_last_combination_in_current_state_combinations(
                                                            last_state_combination,
                                                            state_combinations) 
        while (valid_combination != True) :                                                                                    
            index_sorted = np.argsort(summed_power_of_each_combination)
            summed_power_of_each_combination_sorted = summed_power_of_each_combination[index_sorted]
            
            index, residual = self.find_nearest_original_co(summed_power_of_each_combination_sorted, 
                                                            test_value, 
                                                            index_sorted, 
                                                            summed_power_of_each_combination,
                                                            priority_combo)
            
            #check if it is a valid state combination choice the one that was found
            #Even though we have already constrained the possible appliances (only appliances
            #seen in tha last combo and appliances in current valid_locations), we still
            #need to check if the found combo is a valid one since we could have turned off
            #a device (from last combo) without having the corresponding valid location. 
            #We make this check only of we have location information for this instant.
            if len(valid_locations) == 0:
                valid_combination = True
            else:
                valid_combination, locations_of_changes = self.check_if_valid_state_combination(
                                                            state_combinations[index], 
                                                            last_state_combination, 
                                                            valid_locations,
                                                            appliances_order,
                                                            location_data,
                                                            last_order_of_appliances)
            
            if (first_guess == -1):
                first_guess = index
                
            if (valid_combination != True):
                self.location_used += 1
                summed_power_of_each_combination[index] = -1.0 # we make it invalid
                print ('Looking again...')
                if (abs(residual) > tolerance_margin):
                    index = first_guess
                    valid_combination = True
            
        return index, residual
   
           
    def find_nearest(self, test_array, location_data, vampire_power, resample_seconds):
        appliances_order = self.get_ordering_of_appliances_in_state_combinations_table()
        combos = np.zeros(shape=(1, len(appliances_order)+1))
        residuals = []
        
        #TODO first combination known
        #Everything is possible for the first one
        last_state_combination = -1
        #last_combination_appliances = location_data.appliances_location.keys()
        last_combination_appliances = [7,8,9,17,18]
        #last_order_of_appliances = appliances_order
        last_order_of_appliances = [7,8,9,17,18]
        
        for i, test_value in enumerate(test_array.values):            
            
            valid_locations = self.get_locations_of_events_within_timespan(
                                                                    location_data, 
                                                                    test_array.index[i], 
                                                                    max_time_difference=resample_seconds)
                                                                           
                                                                         
            state_combinations, summed_power_of_each_combination, order_of_appliances = self.get_constrained_state_combinations(
                                                                    valid_locations, 
                                                                    last_combination_appliances, 
                                                                    location_data, 
                                                                    vampire_power)
            
            print(i,'========', 
            '\nNo. combos:', str(len(state_combinations)), 
            '\nValid locs:', str(valid_locations), 
            '\nOrder of   apps:', str(order_of_appliances), 
            '\nLast order apps:', str(last_order_of_appliances),
            '\nLast combo apps:', str(last_combination_appliances),
            '\nLast combo:')
            pp.pprint(last_state_combination)
            
            index, residual = self.find_nearest_location_constraint(
                                                                    summed_power_of_each_combination, 
                                                                    test_value, 
                                                                    location_data, 
                                                                    state_combinations,
                                                                    last_state_combination,
                                                                    order_of_appliances,
                                                                    valid_locations,
                                                                    vampire_power,
                                                                    last_order_of_appliances)
            
            full_combo = self.get_full_state_combination(state_combinations[index], order_of_appliances)
            full_combo = np.hstack((full_combo,vampire_power))            
            combos = np.vstack((combos, full_combo))
            residuals.append(residual)
            
            last_state_combination = np.copy(state_combinations[index])
            last_combination_appliances = self.get_appliances_in_state_combination(order_of_appliances, last_state_combination, location_data) 
            last_order_of_appliances = order_of_appliances
            gc.collect()
        
        combos = np.delete(combos, (0), axis=0)
        return combos, residuals


    def disaggregate(self, mains, output_datastore, location_data=None, mains_values=None, baseline=None, **load_kwargs):

        from sklearn.utils.extmath import cartesian
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Get centroids
        centroids = [model['states'] for model in self.model]
        state_combinations = cartesian(centroids)

        try:
            timezone = location_data.dataset.metadata.get('timezone')
        except Exception:
            timezone = ''

        vampire_power = baseline
        if baseline is None:
            vampire_power = mains.vampire_power() #- correction
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))
        print("vampire_power = {} watts".format(vampire_power))        
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
        
        self.vampire_power = vampire_power
        self.state_combinations_all = state_combinations
        self.summed_power_of_each_combination_all = summed_power_of_each_combination

                
        resample_seconds = load_kwargs.pop('resample_seconds', 60)
        load_kwargs.setdefault('resample', True)
        load_kwargs.setdefault('sample_period', resample_seconds)
        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = '{}/elec/meter1'.format(building_path)

        if mains_values is None:
            load_kwargs['sections'] = load_kwargs.pop('sections', mains.good_sections())
            mains_values = mains.power_series(**load_kwargs)
            using_series = False
        else:
            mains_values = [mains_values]
            using_series = True
        
        self.mains_used = mains_values        
        
        self.location_used = 0
        self.location_loop = 0
        self.co_indices_original = []
        self.co_indices_location = [] #No longer applies since indices constantly change after each iteration. We now return the combo
        self.co_residuals_original = []
        self.co_residuals_location = []
        self.co_combos_location = []
        for chunk in mains_values:


            # Record metadata
            if using_series:
                timeframes.append(TimeFrame(start=chunk.index[0], end=chunk.index[-1]))
                measurement = ('power', 'apparent')
            else:
                timeframes.append(chunk.timeframe)
                measurement = chunk.name

            # Start disaggregation
            print('Calculating original indices of state combinations...')
            indices_of_state_combinations_original, residuals_power_original = find_nearest(
            summed_power_of_each_combination, chunk.values)            
            
            self.co_indices_original.extend(indices_of_state_combinations_original)
            self.co_residuals_original.extend(residuals_power_original)
            
            print('Calculating indices of state combinations...')
            state_combinations_location, residuals_power_location = self.find_nearest(
            chunk, location_data, vampire_power, resample_seconds)
            
            self.co_combos_location.extend(state_combinations_location)
            self.co_residuals_location.extend(residuals_power_location)
            
            #Write results
            for i, model in enumerate(self.model):
                print("Estimating power demand for '{}'".format(model['training_metadata']))
                predicted_power = state_combinations_location[:, i].flatten()
                cols = pd.MultiIndex.from_tuples([measurement])
                meter_instance = model['training_metadata'].instance()
                output_datastore.append('{}/elec/meter{}'
                                        .format(building_path, meter_instance),
                                        pd.DataFrame(predicted_power,
                                                     index=chunk.index,
                                                     columns=cols))

            # Copy mains data to disag output
            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))
        
        
        ##################################
        # Add metadata to output_datastore
        self.add_metadata(output_datastore, measurement, timeframes, mains, timezone, load_kwargs)

    def add_metadata(self, output_datastore, measurement, timeframes, mains, timezone, load_kwargs):


        date_now = datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_CO_' + date_now)        
        resample_seconds = load_kwargs.pop('resample_seconds', 60)        
        
        building_path = '/building{}'.format(mains.building())
        mains_data_location = '{}/elec/meter1'.format(building_path)
        
        # DataSet and MeterDevice metadata:
        meter_devices = {
            'CO': {
                'model': 'CO',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': resample_seconds,
                'max_sample_period': resample_seconds,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=resample_seconds)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        dataset_metadata = {'name': output_name, 'date': date_now,
                            'meter_devices': meter_devices,
                            'timeframe': total_timeframe.to_dict(),
                            'timezone': timezone}
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict(),
                    'good_sections': list_of_timeframe_dicts(merged_timeframes)
                }
            }
        }

        # Appliances and submeters:
        appliances = []
        for model in self.model:
            meter = model['training_metadata']

            meter_instance = meter.instance()

            for app in meter.appliances:
                meters = app.metadata['meters']
                appliance = {
                    'meters': [meter_instance], 
                    'type': app.identifier.type,
                    'instance': app.identifier.instance
                }
                appliances.append(appliance)

            elec_meters.update({
                meter_instance: {
                    'device_model': 'CO',
                    'submeter_of': 1,
                    'data_location': ('{}/elec/meter{}'
                                      .format(building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict(),
                        'good_sections': list_of_timeframe_dicts(merged_timeframes)
                    }
                }
            })

            #Setting the name if it exists
            if meter.name:
                if len(meter.name)>0:
                    elec_meters[meter_instance]['name'] = meter.name

        building_metadata = {
            'instance': mains.building(),
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)



