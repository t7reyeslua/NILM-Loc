# -*- coding: utf-8 -*-
import nilmtk
from nilmtk import DataSet
import pandas as pd
from pandas import *
import numpy as np
import copy
from pandas import DataFrame, Series, DateOffset
import cPickle as pickle
#import sys
#mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
#sys.path.append(mypath)

import preprocessing.smooth as smooth
import settings as settings
from metadata.metadata import Metadata

class LocationInference(object):
    """
    Contains different functions to help infer user occupancy behavior from the
    ECO/REDD Dataset. The output is intended to also be used as an extra constraint for 
    energy disaggregation with the hope to improve accuracy.
    
    We infer the location of the user from the plug data. For example, if the plug
    data tells us that the 'tv' is being used, then it is most certainly that a 
    user is in the same room as the TV.
    
    @author: Antonio Reyes Lúa - TU Delft 2015
    """
          
    minimum_timespan_threshold = 30 #seconds
          
    def __init__(self, dataset_name, smoothen=True, h5_path=None, loc_path=None, house_number=1):  
        """
        Initialize the object depending on the dataset being used.
        Probably good idea to implement being able to choose the house number
        and probably even pass the min_power_threshold as parameter.        
        """
        self.init_values()
        self.smoothed = smoothen
        self.name = dataset_name
        self.metadata = Metadata(dataset_name)
        if dataset_name == 'REDD':
            if h5_path is None:
                h5_path = settings.h5_redd            
            self.dataset = DataSet(h5_path)        
            self.elec = self.dataset.buildings[house_number].elec
        elif dataset_name == 'ECO':
            if h5_path is None:
                h5_path = settings.h5_eco
            self.dataset = DataSet(h5_path)        
            self.elec = self.dataset.buildings[house_number].elec
        elif dataset_name == 'SMART':
            print(dataset_name + ' not yet implemented')
        elif dataset_name == 'iAWE':
            print(dataset_name + ' not yet implemented')
        else:
            print('Invalid datasetname')
            
        if loc_path is not None:
            self.read_object(loc_path)
            
    def init_values(self):
        self.name = None
        self.appliances_ON_times = None
        self.appliances_consuming_times = None
        self.events = None
        self.events_consuming = None
        self.appliances_transitions = None
        self.triggers = None
        self.appliances_triggers = None
        self.appliances_triggers_reduced = None
        self.events_locations = None
        self.locations = None
        self.concurrent_users = None
        self.appliances_status = None
        self.count_appliances_ON_OFF = None
        self.count_appliances_consumption_events = None
        self.count_user_interactions_ON_OFF = None
        self.count_user_interactions = None
        self.events_apps_1min = None
        self.events_apps_1day = None
        self.minutes_with_events = None
        self.daily_events = None
        self.daily_events_per_app = None
        return
    
    def save_object(self, fnpath):
        objects = {}
        
        objects['name'] = self.name
        objects['appliances_ON_times'] = self.appliances_ON_times
        objects['appliances_consuming_times'] = self.appliances_consuming_times
        objects['events'] = self.events
        objects['events_consuming'] = self.events_consuming
        objects['appliances_transitions'] = self.appliances_transitions
        objects['triggers'] = self.triggers
        objects['appliances_triggers'] = self.appliances_triggers
        objects['appliances_triggers'] = self.appliances_triggers
        objects['events_locations'] = self.events_locations
        objects['locations'] = self.locations
        objects['concurrent_users'] = self.concurrent_users
        objects['appliances_status'] = self.appliances_status
        objects['count_appliances_ON_OFF'] = self.count_appliances_ON_OFF
        objects['count_appliances_consumption_events'] = self.count_appliances_consumption_events
        objects['count_user_interactions_ON_OFF'] = self.count_user_interactions_ON_OFF
        objects['count_user_interactions'] = self.count_user_interactions
        objects['events_apps_1min'] = self.events_apps_1min
        objects['events_apps_1day'] = self.events_apps_1day
        objects['minutes_with_events'] = self.minutes_with_events
        objects['daily_events'] = self.daily_events
        objects['daily_events_per_app'] = self.daily_events_per_app        
            
        pickle.dump( objects, open(fnpath + self.name + '_location.p', "wb" ) )        
        return
        
    def read_object(self, fnpath):
        print('Reading LocationInference object from: ' + fnpath)
        objects = pickle.load(open(fnpath + self.name + '_location.p', "rb"))
        self.dismantle_object(objects)
        return objects
        
    def dismantle_object(self, objects):        
        self.name = objects['name'] 
        self.appliances_ON_times = objects['appliances_ON_times'] 
        self.appliances_consuming_times = objects['appliances_consuming_times'] 
        self.events = objects['events'] 
        self.events_consuming = objects['events_consuming'] 
        self.appliances_transitions = objects['appliances_transitions'] 
        self.triggers = objects['triggers'] 
        self.appliances_triggers = objects['appliances_triggers'] 
        self.appliances_triggers_reduced = objects['appliances_triggers_reduced'] 
        self.events_locations = objects['events_locations'] 
        self.locations = objects['locations'] 
        self.concurrent_users = objects['concurrent_users'] 
        self.appliances_status = objects['appliances_status'] 
        self.count_appliances_ON_OFF = objects['count_appliances_ON_OFF'] 
        self.count_appliances_consumption_events = objects['count_appliances_consumption_events'] 
        self.count_user_interactions_ON_OFF = objects['count_user_interactions_ON_OFF'] 
        self.count_user_interactions = objects['count_user_interactions'] 
        self.events_apps_1min = objects['events_apps_1min'] 
        self.events_apps_1day = objects['events_apps_1day'] 
        self.minutes_with_events = objects['minutes_with_events'] 
        self.daily_events = objects['daily_events'] 
        self.daily_events_per_app = objects['daily_events_per_app']  
        print('LocationInference object ready.')       
        return
    
    def calculate_ON_times(self):
        """
        For each appliance being read, calculate the times at which it was 
        being used. It uses the defined threshold for each appliance.
        """
        
        self.appliances_ON_times = {}
        for i in range(self.metadata.min_index, self.metadata.max_index):
            elecmeter = self.elec[i]
            min_power = self.metadata.min_power_threshold[i]
            self.appliances_ON_times[i] = list(elecmeter.when_on(on_power_threshold=min_power))[0]
        
#        if self.name == 'REDD':
#            sm3 = smooth.smooth(list(self.elec[3].power_series())[0], window_len=200)
#            self.appliances_ON_times[3] = smooth.when_on(sm3, on_power_threshold=20)
#            
#            sm4 = smooth.smooth(list(self.elec[4].power_series())[0], window_len=200)
#            self.appliances_ON_times[4] = smooth.when_on(sm4, on_power_threshold=20)
#            
#            sm6 = smooth.smooth(list(self.elec[6].power_series())[0], window_len=200)
#            self.appliances_ON_times[6] = smooth.when_on(sm6, on_power_threshold=20)
#                        
#            sm11 = smooth.smooth(list(self.elec[11].power_series())[0], window_len=80)
#            self.appliances_ON_times[11] = smooth.when_on(sm11, on_power_threshold=20)
#            
#            sm16 = smooth.smooth(list(self.elec[16].power_series())[0], window_len=200)
#            self.appliances_ON_times[16] = smooth.when_on(sm16, on_power_threshold=20)
#            
#            #TODO 10, 20
            
        self.calculate_fridge_user_interactions()
        return self.appliances_ON_times

    
    def calculate_consuming_times(self):
        """
        For each appliance being read, calculate the times at which it was 
        consuming power. It uses the defined threshold for each appliance.
        """
        
        self.appliances_consuming_times = {}
        for i in range(self.metadata.min_index, self.metadata.max_index):
            elecmeter = self.elec[i]
            min_power = self.metadata.min_power_consuming_threshold[i]
            self.appliances_consuming_times[i] = list(elecmeter.when_on(on_power_threshold=min_power))[0]
        
        return self.appliances_consuming_times
    
    def calculate_fridge_user_interactions(self):
        if self.name == 'REDD':            
            fridge_id = 5
        elif self.name == 'ECO':            
            fridge_id = 7
        
        lf = list(self.elec[fridge_id].power_series())[0]    
        lv = 0
        nv = Series(-1, index=lf.index)
        for i,v in enumerate(lf):
            if (v-lv)>200:
                try:
                    for k in range (i, i+25):
                        nv[k] = 200
                except Exception:
                    continue
            else:
                if nv[i] == -1:
                    if v > 420:
                        v -= 220
                    nv[i] = v
            lv = v
        
        nnv = Series(-1, index=lf.index)
        for i,v in enumerate(nv):
            if v>150:
                v -=200
                if v<6:
                    v = 6
            nnv[i] = v
        
        lfon = nnv >=25
        
        self.fridge_nv = nv
        self.fridge_nnv = nnv
        self.fridge_on = lfon
        self.appliances_ON_times[fridge_id] = lfon
            
    def calculate_events(self):
        """
        Get only the timestamp of when each ON/OFF event for each appliance 
        was triggered.
        """
        
        # Detect the moment at which there was a transition for each appliance
        # For this, we shift the whole vector and compare when two consecutive values are different.
        self.events = {}
        for i in self.appliances_ON_times:
            current_state = self.appliances_ON_times[i]
            previous_state = current_state.shift()
            trigger_events = current_state[current_state != previous_state]
            self.events[i] = trigger_events   
        
        if self.name == 'REDD' and self.smoothed:
            adjust = {3:(200,20), 4:(200,20), 6:(200,20), 11:(80,20), 16:(200,20)}
            
            for app in adjust:
                wl = adjust[app][0]
                pt = adjust[app][1]
                
                sm = smooth.smooth(list(self.elec[app].power_series())[0], window_len=wl)
                smWO = smooth.when_on(sm, on_power_threshold=pt)
                smEV = smooth.calculate_events(smWO)            
                smFE = smooth.fit_events(smEV,self.events[app])
                self.events[app] = smFE
                self.appliances_ON_times[app] = smWO
            
        return self.events
    
        
    def calculate_consuming_events(self):
        """
        Get only the timestamp of when each ON/OFF event for each appliance 
        was triggered.
        """
        
        # Detect the moment at which there was a transition for each appliance
        # For this, we shift the whole vector and compare when two consecutive values are different.
        self.events_consuming = {}
        for i in self.appliances_consuming_times:
            current_state = self.appliances_consuming_times[i]
            previous_state = current_state.shift()
            trigger_events = current_state[current_state != previous_state]
            self.events_consuming[i] = trigger_events   
        
        return self.events_consuming
        
        
    def calculate_triggers(self):
        """
        Merge the lists from the output of calculate_ON_times into a single one.
        The new list only contains the transition events including their
        timestamps, appliance that triggered it.
        """
        self.appliances_transitions = DataFrame(copy.deepcopy(self.events))
        self.appliances_transitions = self.appliances_transitions.fillna(0)
        self.triggers = copy.deepcopy(self.events)
        # Replace all TRUE/FALSE with the index of the appliance
        for i in self.triggers:            
            self.triggers[i][self.triggers[i] == False] = True
            self.triggers[i] = self.triggers[i] * i
        
        # Merge all appliances into a single list and replace NaN with 0
        self.appliances_triggers = DataFrame(self.triggers)
        self.appliances_triggers = self.appliances_triggers.fillna(0)
        
        # Delete first row since it will always be detected as a transition for all appliances
        self.appliances_triggers = self.appliances_triggers.ix[1:] 
        
        # Reduce DataFrame to single column series containing a list of only the triggered appliances
        # i.e. get rid of all the cells that contain zeroes.
        reduced_list = {}
        for i in self.appliances_triggers.index:
            reduced_list[i] = [x for x in self.appliances_triggers[str(i)].values[0].tolist() if x != 0]
        
        self.appliances_triggers_reduced = Series(reduced_list)
        return self.appliances_triggers_reduced
        
    def calculate_locations(self):
        """
        From the list obtained in calculate_triggers, replace each identified 
        appliance in the list with it corresponding location.
        """
        self.locations = Series(index=self.appliances_triggers_reduced.index)

        for i in self.appliances_triggers_reduced.index:
            l = []
            [l.extend(self.metadata.appliances_location[j]) for j in self.appliances_triggers_reduced[i]]
            s = list(set(l))
            self.locations[i] = s
            
        self.events_locations = DataFrame(self.appliances_triggers_reduced, columns = ['Events'])
        self.events_locations['Locations'] = self.locations
        
        return self.locations
            
        
    def calculate_users(self):
        """
        Determine the number of concurrent users by defining a 
        minimum_timespan_threshold that indicates whether it is possible or not
        for a single user to activate the group of events that fall within that 
        timespan. 
        
        We consider:
        - If n events happen at the same exact moment, then there are n concurrent users,
          even if all the events happen in the same room.
        - If n events happen within the minimum_timespan_threshold, then there are m
          concurrent users, where m is the number of different rooms where events ocurred.
        """
        self.concurrent_users = Series(index=self.appliances_triggers_reduced.index)
        
        for i in range(0, len(self.events_locations.index)):
            # Get all events that fall within +- the minimum thimespan threshold.
            # These events are the ones that might indicate us how many concurrent users there are at that moment.
            concurrent_events = self.events_locations[(self.events_locations.index[i] - DateOffset(seconds = self.minimum_timespan_threshold)):(self.events_locations.index[i] + DateOffset(seconds = self.minimum_timespan_threshold))]
                      
            # Count how many concurrent events ocurred at the same exact moments from all the
            # pool of concurrent events. For example, if two events occur at the same precise instant, then
            # if both hapen in different rooms, both will be accounted when counting rooms,
            # but if both happen in the same room then one of them will not be counted as there
            # will be only 1 room but 2 different events. Therefore, we need to take this into account.
            nevs  = []            
            nlocs = []
            [nevs.append(len(j)) for j in concurrent_events['Events'].values]
            [nlocs.append(len(j)) for j in concurrent_events['Locations'].values]
            users_at_exact_moment = np.sum(np.subtract(nevs, nlocs))
            
            # Count in how many different rooms an event was triggered within the timespan
            locs = []
            [locs.extend(j) for j in concurrent_events['Locations'].values]
            slocs = list(set(locs))
            locations_within_timespan = len(slocs)
            
            # Make the final count
            users = locations_within_timespan + users_at_exact_moment
            self.concurrent_users[i] = users
            
        self.events_locations['Users'] = self.concurrent_users
        
        return self.concurrent_users
            
    def calculate_ground_truth(self):
        gt = DataFrame(copy.deepcopy(self.events_consuming))
        gt = gt.fillna(-1)
        for x in gt.columns:
            l = len(gt[x])
            for y in range(1,l):
                if (gt[x][y] == -1):
                    gt[x][y] = gt[x][y-1]
        self.appliances_status = gt
    
    def count_ON_OFF_events(self):
        counts = {}
        counts_total = {}
        for app in self.events_consuming:
            n_true = len(self.events_consuming[app][self.events_consuming[app]==True])
            n_false = len(self.events_consuming[app][self.events_consuming[app]==False])
            counts[app] = (n_true, n_false)
            counts_total[app] = min(n_true, n_false)
        self.count_appliances_ON_OFF = DataFrame(counts, index=['ON','OFF'])
        self.count_appliances_consumption_events = DataFrame(counts_total, index=['times consuming'])
        
    def count_user_interactions_events(self):
        counts = {}
        counts_total = {}
        for app in self.events:
            n_true = len(self.events[app][self.events[app]==True])
            n_false = len(self.events[app][self.events[app]==False])
            counts[app] = (n_true, n_false)
            counts_total[app] = min(n_true, n_false)
        self.count_user_interactions_ON_OFF = DataFrame(counts, index=['ON','OFF'])
        self.count_user_interactions = DataFrame(counts_total, index=['times used'])
    
    def group_events_apps_1min(self):
        index1m = self.events_locations.resample('1Min').dropna().index
        appliances_lists = []
        locations_lists = []
        for timestamp in index1m:
            concurrent_events = self.events_locations['Locations'][timestamp : (timestamp + DateOffset(seconds = 60))]
            concurrent_appliances = self.events_locations['Events'][timestamp: (timestamp + DateOffset(seconds = 60))]
            locs = []
            [locs.extend(j) for j in concurrent_events.values]
            locations_within_timespan = locs
            apps = []
            [apps.extend(j) for j in concurrent_appliances.values]
            appliances_within_timespan = apps
            locations_lists.append(locations_within_timespan)
            appliances_lists.append(appliances_within_timespan)
        p = Series(locations_lists, index=index1m)
        pp = DataFrame(p, columns=['Locs'])
        pp['Apps'] = appliances_lists
        self.events_apps_1min = pp
    
    
    def group_events_apps_1day(self):
        index1d = self.events_locations.resample('D').dropna().index
        appliances_lists = []
        locations_lists = []
        for timestamp in index1d:
            concurrent_events = self.events_locations['Locations'][timestamp : (timestamp + DateOffset(days = 1))]
            concurrent_appliances = self.events_locations['Events'][timestamp: (timestamp + DateOffset(days = 1))]
            locs = []
            [locs.extend(j) for j in concurrent_events.values]
            locations_within_timespan = locs
            apps = []
            [apps.extend(j) for j in concurrent_appliances.values]
            appliances_within_timespan = apps
            locations_lists.append(locations_within_timespan)
            appliances_lists.append(appliances_within_timespan)
        p = Series(locations_lists, index=index1d)
        pp = DataFrame(p, columns=['Locs'])
        pp['Apps'] = appliances_lists
        self.events_apps_1day = pp    

    def count_minutes_with_events(self):        
        index1d = self.events_locations.resample('D').dropna().index
        
        mins_events = []
        for timestamp in index1d:
            minutes_with_events = len(self.events_apps_1min[timestamp : (timestamp + DateOffset(days = 1))])
            mins_events.append(minutes_with_events)
            
        r = {'Mins with events':mins_events}   
        dd = DataFrame(r, index=index1d)
        self.minutes_with_events = dd
        

    def count_daily_events(self):
        lnlocs = []
        lnapps = []
        lslocs = []
        lsapps = []
        lnlocsd= []
        lnappsd= []
        for i,day in self.events_apps_1day.iterrows():
            nlocs = len(day['Locs'])
            napps = len(day['Apps'])
            slocs = list(set(day['Locs']))
            sapps = list(set(day['Apps']))
            nlocsd= len(slocs)
            nappsd= len(sapps)
            lnlocs.append(nlocs)
            lnapps.append(napps)
            lslocs.append(slocs)
            lsapps.append(sapps)
            lnlocsd.append(nlocsd)
            lnappsd.append(nappsd)
        r = {'nLocs': lnlocs, 'nApps': lnapps, 'nLocsDiff': lnlocsd, 'nAppsDiff': lnappsd, 'Locs': lslocs, 'Apps': lsapps} 
        dd = DataFrame(r, index=self.events_apps_1day.index)
        dd['Mins with events'] = self.minutes_with_events['Mins with events']
        self.daily_events = dd

    def count_daily_events_per_app(self):
        apps = self.metadata.get_channels()
        r = {}
        for app in apps:
            r[app] = []
        for i,day in self.events_apps_1day.iterrows():
            for app in apps:
                n = day['Apps'].count(app)
                r[app].append(n)

        dd = DataFrame(r, index=self.events_apps_1day.index)
        self.daily_events_per_app = dd

    def infer_locations(self):
        self.calculate_ON_times()
        self.calculate_events()
        self.calculate_triggers()
        self.calculate_locations()
        self.calculate_users()
        
        self.calculate_consuming_times()
        self.calculate_consuming_events()
        self.calculate_ground_truth()
        
        self.count_ON_OFF_events()
        self.count_user_interactions_events()
        
        self.group_events_apps_1min()
        self.count_minutes_with_events()        
        self.group_events_apps_1day()
        self.count_daily_events()        
        self.count_daily_events_per_app()
        

                    
                
        
            
        
        
        

        