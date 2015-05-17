# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, DateOffset
import numpy as np 

class GroundTruth(object):
    
    def __init__(self, loc, co, sample_period=60, resample=True, good_sections_only=True):
        self.loc = loc
        self.co = co
        
        self.sample_period = sample_period
        self.resample = resample
        self.good_sections_only = good_sections_only
        
        self.power_series_mains = None
        self.power_series_mains_from_apps = None 
        self.power_series_channels_table = None
        self.power_series_apps_table = None
        self.power_series_channels = None
        self.power_series_apps = None    
        self.power_series_mains_with_timestamp = None
                
        self.vampire_power = None
        self.state_combinations = None
        self.summed_power_of_each_combination = None
        
        self.event_locations = None
        self.event_appliances = None
        self.timestamps = None
        self.mains_values = None
        self.gt_appliances = None
        self.get_appliances_states = None
        self.gt_appliances_summed_power = None
        self.gt_appliances_residual = None
        self.ground_truth_table = None
                
        self.comparison = None
        self.comparison_extended = None
        return
        
    def generate_mains_power_series(self):
        mains = self.loc.elec.mains()
        load_kwargs={}
        load_kwargs.setdefault('resample', self.resample)
        load_kwargs.setdefault('sample_period', self.sample_period)
        if self.good_sections_only is True:
            load_kwargs['sections'] = mains.good_sections()
        
        chunks = list(mains.power_series(**load_kwargs))
        self.power_series_mains = chunks
        return chunks
        
    def generate_apps_power_series(self):
        load_kwargs={}
        load_kwargs.setdefault('resample', self.resample)
        load_kwargs.setdefault('sample_period', self.sample_period)
        
        #Get power series of each channel (elecmeter)
        ps = {}
        for i in  self.loc.min_power_threshold:
            ps[i] = list(self.loc.elec[i].power_series(**load_kwargs))[0]
            
        #Group corresponding channels to get power series of appliance.
        #Some appliances are metergroups themselves (2+ elecmeters)
        pst = dict(ps)
        if self.loc.name == 'REDD':
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
        
        
        gt_ps = DataFrame(ps)
        gt_pst = DataFrame(pst)
                
        self.power_series_channels_table = gt_ps
        self.power_series_apps_table = gt_pst
        
        self.power_series_channels = ps
        self.power_series_apps = pst
        
        return ps, pst
    
        
    def generate_mains_power_series_from_apps(self):
        
        if not all([self.vampire_power, self.timestamps, self.power_series_channels]):
            print ('Run first: generate_gt_values, generate_apps_power_series, generate_state_combinations_all.')
            return
            
        sum_of_apps_power = []
        for timestamp in self.timestamps:
            sum_of_each_app_values = 0
            for app in self.power_series_channels.keys():
                sum_of_each_app_values += self.power_series_channels[app][timestamp]
            sum_of_apps_power.append(sum_of_each_app_values + self.vampire_power)
        
        mains_created = Series(sum_of_apps_power, index=self.timestamps)
        
        fmains = [float(value) for value in self.mains_values]
        smains = Series(fmains, index=self.timestamps)
        
        self.power_series_mains_with_timestamp = smains
        self.power_series_mains_from_apps = mains_created
        return mains_created
        
    def generate_state_combinations_all(self):
        mains = self.loc.elec.mains()
        
        from sklearn.utils.extmath import cartesian
        centroids = [model['states'] for model in self.co.model]
        state_combinations = cartesian(centroids)
        
        correction = 0 #28.35
        vampire_power = mains.vampire_power() - correction
        n_rows = state_combinations.shape[0]
        vampire_power_array = np.zeros((n_rows, 1)) + vampire_power
        state_combinations = np.hstack((state_combinations, vampire_power_array))
        summed_power_of_each_combination = np.sum(state_combinations, axis=1)
                
        self.vampire_power = vampire_power
        self.state_combinations = state_combinations
        self.summed_power_of_each_combination = summed_power_of_each_combination
        return vampire_power, state_combinations, summed_power_of_each_combination
    
        
        
    def generate_gt_values(self):
        
        if not all([self.loc, self.co, self.vampire_power, self.power_series_mains, self.power_series_channels]):
            print ('Run first: generate_mains_power_series, generate_apps_power_series, generate_state_combinations_all.')
            return
            
        locations_lists  = []
        appliances_lists = []
        timestamps_list = []
        mains_values = []
        gt = []
        gt_sums = []
        gt_residuals = []
        gt_states = []
        offset = 60
        for chunk in self.power_series_mains:
            for ts, value in enumerate(chunk):
                timestamp = chunk.index[ts]
                
                #Get all the events that happened in the last minute
                concurrent_events = self.loc.events_locations['Locations'][(timestamp - DateOffset(seconds = offset)):(timestamp)]
                concurrent_appliances = self.loc.events_locations['Events'][(timestamp - DateOffset(seconds = offset)):(timestamp)]
                
                
                gt_appliances = None
                gt_apps = []
                for gt_event_ts in self.loc.appliances_status.index:
                    if gt_event_ts <= timestamp:
                        gt_appliances = self.loc.appliances_status[str(gt_event_ts)]
                        gt_ts = gt_event_ts
                if gt_appliances is not None:
                    gt_apps = [v for i,v in enumerate(gt_appliances) if gt_appliances.values[0][i] == True]  
                
                
                if (len(gt_apps) == 0):
                    gt.append([])
                    gt_sums.append(0)
                    gt_residuals.append(0)
                    gt_states.append([])
                else:
                    gt_state_combinations, summ, order_of_appliances = self.get_gt_state_combinations(
                                                                                gt_apps,
                                                                                self.loc, 
                                                                                self.vampire_power,
                                                                                timestamp,
                                                                                self.power_series_channels, 
                                                                                self.co)
                    
                    gt_apps1 = [v[0] for v in gt_state_combinations if v[1] not in (0, self.vampire_power)]
        
                    gt.append(gt_apps1)
                    gt_sums.append("{0:.2f}".format(summ))
                    gt_residuals.append("{0:.2f}".format((summ-value)))
                     
                    gt_sc = [int(v[1]) for v in gt_state_combinations if v[1] not in (0, self.vampire_power)]
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
        gt_sums[0] = gt_sums[1]
        
        self.event_locations = locations_lists
        self.event_appliances = appliances_lists
        self.timestamps = timestamps_list
        self.mains_values = mains_values
        self.gt_appliances = gt
        self.get_appliances_states = gt_states
        self.gt_appliances_summed_power = gt_sums
        self.gt_appliances_residual = gt_residuals
        
        vals = {'EvLocations': locations_lists, 'EvAppliances': appliances_lists, 'Mains': mains_values, 'GT apps': gt, 'GT states': gt_states, 'GT sum of apps': gt_sums, 'GT sum residual': gt_residuals}
        self.ground_truth_table = DataFrame(vals,index=timestamps_list)
        return locations_lists, appliances_lists, timestamps_list, mains_values, gt, gt_sums, gt_residuals, gt_states
    
    def find_nearest(self, array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
        
    def get_gt_state_combinations(self, gt_apps, loc, vampire_power, timestamp, gt_power_series, co):
        
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
            
        state_combinations =  [(v, self.find_nearest(centroids_on[v], values[v])) for v in values]   
        values_of_combination = [self.find_nearest(centroids_on[v], values[v]) for v in values] 
        summed_power_of_combination = sum(values_of_combination) + vampire_power
    
        return state_combinations, summed_power_of_combination, ordering

    
    def get_difference(self, serieA, serieB):
        c = []
        for i, gs in enumerate(serieB):
            difference = float(gs) - serieA.values[i]
            c.append(difference)        
        return c
        
    def compare_mains_and_gt_of_appliances(self):
        
        if not all([self.loc, self.co, self.timestamps, self.vampire_power, self.mains_values, self.power_series_mains_from_apps, self.power_series_apps_table]):
            print ('Run first: generate_mains_power_series_from_apps, generate_apps_power_series, generate_state_combinations_all, generate_gt_values.')
            return
            
        sum_of_apps_power = self.power_series_mains_from_apps.values
        comparison_mains_and_apps_abs = []
        comparison_mains_and_apps = []
        for i,summ in enumerate(sum_of_apps_power):
            diff = sum_of_apps_power[i] - float(self.mains_values[i])
            comparison_mains_and_apps_abs.append(abs(diff))
            comparison_mains_and_apps.append(diff)
            
        t = {}
        t['1. summ of apps'] = sum_of_apps_power
        t['2. mains'] = self.mains_values         
        t['3. diff'] = comparison_mains_and_apps
        t['4. diffabs'] = comparison_mains_and_apps_abs
        d = DataFrame(t, index=self.timestamps)
        
        
        gt_all = DataFrame(self.power_series_apps_table)
        gt_all["sum"] = gt_all.sum(axis=1)
        gt_all["mains_apps"] = gt_all['sum'] + self.vampire_power
        gt_all['mains'] = self.mains_values
        gt_all["diff1"] = abs(gt_all['mains_apps'] - gt_all['mains'])
        gt_all["diff2"] = abs(gt_all['sum'] - gt_all['mains'])
    
        self.comparison = d
        self.comparison_extended = gt_all
        return d, gt_all
        
    def generate(self):
        self.generate_state_combinations_all()
        self.generate_mains_power_series()
        self.generate_apps_power_series()
        self.generate_gt_values()
        self.compare_mains_and_gt_of_appliances()
        
        
    def save_to_file(self, fn):
        gg = DataFrame(self.power_series_apps_table)
        try:
            del gg['diff1']
            del gg['diff2']
        except Exception:
            dummy = 0
            
        gg['Loc Events'] = self.loc.events_apps_1min['Apps']
        apps = self.loc.metadata.get_channels()
        sd = {}
        #Initialize series with 0s
        for app in apps:
            sd[app] = Series(0, index=gg.index)
            
        #Count location events for each appliance
        for index, row in gg.iterrows():
            try:
                if len(row['Loc Events']) > 0:
                    for app in apps:
                        n = row['Loc Events'].count(app)
                        sd[app][index] = n
            except Exception:
                continue
        
        if self.loc.name == 'REDD':
            sd[(3,4)] = sd[3]
            sd[(10,20)] = sd[10]
            del sd[3]
            del sd[4]
            del sd[10]
            del sd[20]
          
        #Change column names and append them to gral table
        locevents = DataFrame(sd)
        locevents.columns = [(str(col) + ' locEv') for col in locevents]        
        for locEv in locevents:
            gg[locEv] = locevents[locEv]
            
        
        #Get power values of each appliance and resample for 1min
        act = DataFrame(self.loc.appliances_consuming_times)
        act = act.resample('1Min')
               
        if self.loc.name == 'REDD':
            del act[3]
            del act[10]
            act.columns = [(3,4), 5,6,7,8,9,11,12,13,14,15,16,17,18,19,(10,20)]
        act.columns = [(str(col) + ' conEv') for col in act]
        
        for app in act:
            gg[app] = act[app]        
        gg.columns = [str(col) for col in gg]
        gg = gg[sorted(gg.columns)]
        gg.to_csv(fn)   
        return
        


    
