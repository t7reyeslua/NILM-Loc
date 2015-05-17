"""
Created on Thu May 16 10:43:17 2015

@author: t7
"""

from nilmtk.dataset_converters.eco import convert_eco
from nilmtk.dataset_converters.redd import convert_redd
from nilmtk.dataset_converters.iawe import convert_iawe

import sys
mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
sys.path.append(mypath)

import settings as settings

def get_dataset_names():    
    return settings.dataset_names

def convert_dataset(dataset_name, source, outputhdf5):
    print('Converting...')
    if dataset_name == settings.dataset_names[0]:
        convert_redd(source, outputhdf5)
    elif dataset_name == settings.dataset_names[1]:
        convert_eco(source, outputhdf5, 'CET')
    elif dataset_name == settings.dataset_names[2]:
        print(dataset_name + ' conversion not yet implemented')
    elif dataset_name == settings.dataset_names[3]:        
        convert_iawe(source, outputhdf5)





