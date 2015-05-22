"""
Created on Thu May 16 10:43:17 2015

@author: t7
"""
from nilmtk import HDFDataStore
from nilmtk import DataSet
from nilmtk.dataset_converters.eco import convert_eco
from nilmtk.dataset_converters.redd import convert_redd
from nilmtk.dataset_converters.iawe import convert_iawe
from pandas import DataFrame, Series

#import sys
#mypath = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Code/NILM-Loc'
#sys.path.append(mypath)

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

def disaggregate_original_co(h5_input, h5_output,dataset_start_date_disag, dataset_end_date_disag, centroids=None ):
    import nilmtk.disaggregate as original_nilmtk
    ds = DataSet(h5_input)
    elec = ds.buildings[1].elec
    
    vampire_power_used_in_original = elec.mains().vampire_power()

    #Train
    plain_co = original_nilmtk.CombinatorialOptimisation()
    plain_co.train(elec)
    
    #Modify centroids manually
    if centroids is not None:            
        for i, model in enumerate(plain_co.model):
            instance = model['training_metadata'].instance()
            model['states'] = centroids[instance]
    
    
    #Disaggregate
    ds.set_window(start=dataset_start_date_disag, end=dataset_end_date_disag)
    elec = ds.buildings[1].elec
    output_plain_co = HDFDataStore(h5_output, 'w')
    plain_co.disaggregate(elec.mains(), output_plain_co)
    output_plain_co.close()
       
    return plain_co, vampire_power_used_in_original

def get_disaggregation_predictions(disag_elec, vampire_power, start_date=None, end_date=None):
    if start_date is None:
        start_date = str(disag_elec.get_timeframe().start)
    if end_date is None:
        end_date = str(disag_elec.get_timeframe().end)
    
        
    lm = list(disag_elec.mains().power_series())[0][start_date:end_date]
    mts = {}
    results = []
    summ_of_predicted_apps = []
    mains = []
    for i, submeter in enumerate(disag_elec.submeters().meters):
        instance = disag_elec.submeters().meters[i].instance()
        mts[instance] = list(disag_elec.submeters().meters[i].power_series())[0][start_date:end_date]
    for ts in lm.index:
        values = []
        summ = 0
        for submeter in mts:
            v = mts[submeter][ts]
            if v > 0:
                summ += v
                values.append((submeter, v))
        results.append(values)
        summ_of_predicted_apps.append(summ)
    
    
    summ_of_apps_and_vampire = []
    for i in summ_of_predicted_apps:
        summ_of_apps_and_vampire =  "{0:.2f}".format(i + vampire_power)
    
    r = {'Predicted Estimations': results, 'Sum of predictions': summ_of_predicted_apps,
         'incl. vampire': summ_of_apps_and_vampire}
    res = DataFrame(r, index=lm.index)
    return res
    
def plot(elec, meters_id, start_date, end_date):
    from pylab import rcParams
    import matplotlib.pyplot as plt
    
    rcParams['figure.figsize'] = (14, 6)
    plt.style.use('ggplot')
    power_series = {}
    
    
    for meter in meters_id:
        tag = elec[meter].appliances[0].label(pretty=True) + ', ' + str(meter)
        power_series[tag] = list(elec[meter].power_series())[0]
    
    df = DataFrame(power_series)
    df[start_date:end_date].plot()
    
    return df

def plot_with_mains(elec, meters_id, start_date, end_date, mains=None):
    from pylab import rcParams
    import matplotlib.pyplot as plt
    
    rcParams['figure.figsize'] = (14, 6)
    plt.style.use('ggplot')
    power_series = {}
    
    if mains is None:
        mains = list(elec.mains().power_series())[0]
    power_series['mains'] = mains
    
    for meter in meters_id:
        tag = elec[meter].appliances[0].label(pretty=True) + ', ' + str(meter)
        power_series[tag] = list(elec[meter].power_series())[0]
    
    df = DataFrame(power_series)
    
    df[start_date:end_date].plot()
    
    return df