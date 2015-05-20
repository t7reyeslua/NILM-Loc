# -*- coding: utf-8 -*-
#import sys
#mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
#sys.path.append(mypath)

import numpy as np
#import settings as settings

appliances_location_eco = {4:['hall'],          #tablet computer charger
                           5:['kitchen'],       #dish washer
                           6:['kitchen'],       #air handling unit
                           7:['kitchen'],       #fridge
                           8:['living room'],   #HTPC
                           9:['kitchen'],       #freezer
                           10:['kitchen'],      #kettle
                           11:['bedroom'],      #lamp
                           12:['bedroom'],      #laptop
                           13:['kitchen'],      #stove
                           14:['living room'],  #television
                           15:['living room']   #audio system
                               }                            
        
#TODO decide correct location for each lightning in REDD
appliances_location_redd ={3:['kitchen'],       #oven
                           4:['kitchen'],       #oven
                           5:['kitchen'],       #fridge
                           6:['kitchen'],       #dishwasher
                           7:['kitchen'],       #kitchen outlets
                           8:['kitchen'],       #kitchen outlets
                           9:['bedroom'],       #lightning                
                           10:['bathroom'],     #washer dryer
                           11:['kitchen'],      #microwave
                           12:['bathroom'],     #bathroom_gfi
                           13:['living room'],  #electric heat
                           14:['kitchen'],      #stove
                           15:['kitchen'],      #kitchen outlets
                           16:['kitchen'],      #kitchen outlets                  
                           17:['bathroom'],     #lightning 
                           18:['living room'],  #lightning 
                           19:['bathroom'],     #washer dryer
                           20:['bathroom']      #washer dryer
                           }

min_power_threshold_eco = {4:100,   #tablet computer charger - Not really useful since always charging, 
                           5:1000,  #dish washer
                           6:20,    #air handling unit
                           7:1100,  #fridge
                           8:20,    #HTPC
                           9:1100,  #freezer
                           10:300,  #kettle
                           11:10,   #lamp
                           12:10,   #laptop
                           13:20,   #stove
                           14:70,   #television
                           15:20    #audio system
                           }
                           
#TODO recheck values for some appliances. Some have too many transitions
min_power_threshold_redd ={3:1600,  #oven
                           4:2200,  #oven
                           5:2500,  #fridge 
                           6:30,    #dishwasher
                           7:2300,  #kitchen outlets
                           8:55,    #kitchen outlets
                           9:50,    #lightning - mostly ON during the night         
                           10:100,  #washer dryer
                           11:50,   #microwave
                           12:30,   #bathroom_gfi
                           13:100,  #electric heat
                           14:1200, #stove
                           15:1000, #kitchen outlets
                           16:1400, #kitchen outlets                  
                           17:55,   #lightning - mostly ON during the afternoon
                           18:5,    #lightning - mostly ON during a short period before midnight
                           19:25,   #washer dryer
                           20:100   #washer dryer
                           }
   
#These are practically the same as the ones above. The main difference is that the ones 
#above are intended to describe user interactions and these are for describing when an
#appliance is consuming power even without user intervention. This is the case of the
#fridge and of kitchen outlets 7,8 that are always drawing power.
min_power_consuming_threshold_redd ={
                           3:1600,  #oven
                           4:2200,  #oven
                           5:25,    #fridge 
                           6:200,   #dishwasher
                           7:10,    #kitchen outlets
                           8:10,    #kitchen outlets
                           9:50,    #lightning - mostly ON during the night         
                           10:100,  #washer dryer
                           11:50,   #microwave
                           12:30,   #bathroom_gfi
                           13:100,  #electric heat
                           14:1200, #stove
                           15:1000, #kitchen outlets
                           16:1400, #kitchen outlets                  
                           17:55,   #lightning - mostly ON during the afternoon
                           18:5,    #lightning - mostly ON during a short period before midnight
                           19:25,   #washer dryer
                           20:100   #washer dryer
                           }

min_power_consuming_threshold_eco = {
                           4:100,   #tablet computer charger - Not really useful since always charging, 
                           5:1000,  #dish washer
                           6:20,    #air handling unit
                           7:1100,  #fridge
                           8:20,    #HTPC
                           9:1100,  #freezer
                           10:300,  #kettle
                           11:10,   #lamp
                           12:10,   #laptop
                           13:20,   #stove
                           14:70,   #television
                           15:20    #audio system
                           }                           
                           
user_dependent_appliances_redd = [9,12,17,18]
user_dependent_appliances_eco  = [8,11,12,14,15]
                           
centroids_redd3 = {5         :np.array([0, 49, 198]), #fridge
                  6         :np.array([0, 1075]),    #dishwasher
                  7         :np.array([14,21]),      #kitchen outlets
                  8         :np.array([22,49,78]),   #kitchen outlets
                  9         :np.array([0,41,82]),    #lightning
                  11        :np.array([0,1520]),     #microwave
                  12        :np.array([0,1620]),     #bathroom_gfi
                  13        :np.array([0,15]),       #electric heater
                  14        :np.array([0,1450]),     #stove
                  15        :np.array([0, 1050]),    #kitchen outlets
                  16        :np.array([0,1450]),     #kitchen outlets
                  17        :np.array([0,65]),       #lightning
                  18        :np.array([0,45,60]),    #lightning
                  19        :np.array([0]),          #lightning
                  (3, 4)    :np.array([0,4200]),     #electric oven
                  (10, 20)  :np.array([0,3750])      #washer dryer
                  }

centroids_redd = {5         :np.array([0, 198]),     #fridge
                  6         :np.array([0, 1075]),    #dishwasher
                  7         :np.array([14,21]),      #kitchen outlets
                  8         :np.array([22,78]),      #kitchen outlets
                  9         :np.array([0, 82]),      #lightning
                  11        :np.array([0,1520]),     #microwave
                  12        :np.array([0,1620]),     #bathroom_gfi
                  13        :np.array([0,15]),       #electric heater
                  14        :np.array([0,1450]),     #stove
                  15        :np.array([0, 1050]),    #kitchen outlets
                  16        :np.array([0,1450]),     #kitchen outlets
                  17        :np.array([0,65]),       #lightning
                  18        :np.array([0, 60]),      #lightning
                  19        :np.array([0]),          #lightning
                  (3, 4)    :np.array([0,4200]),     #electric oven
                  (10, 20)  :np.array([0,3750])      #washer dryer
                  }
                  
#TODO centroids_eco
centroids_eco = {}


class Metadata(object):

    def __init__(self, dataset_name):
        self.name = dataset_name
        self.min_index = 0
        self.max_index = 0
        self.min_power_threshold = {}
        self.min_power_consuming_threshold = {}
        self.appliances_location = {}
        self.centroids = {}
        self.user_dependent_appliances = []
        if dataset_name == 'REDD': 
            self.min_index = min(appliances_location_redd)
            self.max_index = max(appliances_location_redd) + 1
            self.min_power_threshold = min_power_threshold_redd
            self.min_power_consuming_threshold = min_power_consuming_threshold_redd
            self.appliances_location = appliances_location_redd
            self.user_dependent_appliances = user_dependent_appliances_redd
            self.centroids = centroids_redd            
        elif dataset_name == 'ECO': 
            self.min_index = min(appliances_location_eco)
            self.max_index = max(appliances_location_eco) + 1
            self.min_power_threshold = min_power_threshold_eco
            self.min_power_consuming_threshold = min_power_consuming_threshold_eco
            self.appliances_location = appliances_location_eco
            self.user_dependent_appliances = user_dependent_appliances_eco
            self.centroids = centroids_eco
        elif dataset_name == 'SMART': 
            print(dataset_name + ' metadata not yet implemented')
        elif dataset_name == 'iAWE':  
            print(dataset_name + ' metadata not yet implemented')

    def get_apps(self):
        apps = self.appliances_location.keys()
        if self.name == 'REDD':
            apps.remove(3)
            apps.remove(4)
            apps.remove(10)
            apps.remove(20)
            apps.append((3,4))
            apps.append((10,20))
        return apps
        
    def get_channels(self):
        apps = self.appliances_location.keys()
        return apps

