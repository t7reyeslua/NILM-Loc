# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:00:30 2015

@author: t7
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:43:17 2015

@author: t7
"""

from nilmtk.dataset_converters import convert_eco
from nilmtk import DataSet

source = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/ETHZ_ECO/house_2'
outputhdf5 = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/ETHZ_ECO/house_2/eco-h2.h5'

print("Converting...")
convert_eco(source, outputhdf5, 'CET')
eco = DataSet(outputhdf5)
eco.set_window(start=None, end='2012-06-08 00:00:00')

