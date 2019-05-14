#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import ensclus_lib as ecl

### Reading inputs from input file
if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_CLUStool.in'

keys = 'INPUT_PATH genfilenames dir_OUTPUT exp_name timestep level season area extreme numclus perc numpcs field_to_plot n_color_levels n_levels draw_contour_lines overwrite_output clim_compare obs_compare climat_file obs_file cmap cmap_cluster cb_label fig_format max_ens_in_fig check_best_numclus fig_ref_to_obs taylor_w_numbers flag_perc'
keys = keys.split()
itype = [str, str, str, str, str, float, str, str, str, int, float, int, str, int, int, bool, bool, bool, bool, str, str, str, str, str, str, int, bool, bool, bool, bool]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['n_color_levels'] = 21
defaults['n_levels'] = 5
defaults['draw_contour_lines'] = False
defaults['field_to_plot'] = 'anomalies'
defaults['overwrite_output'] = False
defaults['run_compare'] = False
defaults['cmap'] = 'RdBu_r'
defaults['cmap_cluster'] = 'nipy_spectral'
defaults['fig_format'] = 'pdf'
defaults['max_ens_in_fig'] = 30
defaults['check_best_numclus'] = False
defaults['fig_ref_to_obs'] = False
defaults['taylor_w_numbers'] = True
defaults['flag_perc'] = True

inputs = ecl.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults, verbose = True)

OUTPUTdir = inputs['dir_OUTPUT'] + ecl.std_outname(inputs['exp_name'], inputs) + '/'

# Creating OUTPUT directory
if not os.path.exists(OUTPUTdir):
    os.mkdir(OUTPUTdir)
    print('The output directory {0} is created'.format(OUTPUTdir))
else:
    print('The output directory {0} already exists'.format(OUTPUTdir))
    if inputs['overwrite_output']:
        print('Overwrite the results in the output directory {0}'.format(OUTPUTdir))
    else:
        raise ValueError('overwrite_output is False and exp_name is already used. Change exp_name or switch overwrite_output to True.')

# open our log file
# logname = 'log_EnsClus_{}.log'.format(datestamp())
# logfile = open(logname,'w') #self.name, 'w', 0)
#
# # re-open stdout without buffering
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#
# # redirect stdout and stderr to the log file opened above
# os.dup2(logfile.fileno(), sys.stdout.fileno())
# os.dup2(logfile.fileno(), sys.stderr.fileno())

#____________Building the array of file names

filgenname = inputs['genfilenames']
modcart = inputs['INPUT_PATH']
namfilp = filgenname.split('*')
if '/' in filgenname:
    modcart = inputs['INPUT_PATH'] + '/'.join(filgenname.split('/')[:-1]) + '/'
    namfilp = filgenname.split('/')[-1].split('*')

lista_all = os.listdir(modcart)
lista_oks = [modcart + fi for fi in lista_all if np.all([namp in fi for namp in namfilp])]

filenames = np.sort(lista_oks)

print('\n***************************INPUT*************')
print('Input file names contain the string: {0}'.format(inputs['genfilenames']))
print('_____________________________\nARRAY OF {0} INPUT FILES:'.format(len(filenames)))
for i in filenames:
    print(i)
print('_____________________________\n')


inputs['filenames'] = filenames
inputs['numens'] = len(filenames)
inputs['OUTPUTdir'] = OUTPUTdir

#____________Building the name of output files
inputs['name_outputs'] = ecl.std_outname(inputs['exp_name'], inputs)

# devo ancora leggere i filez e passarli a ensclus
OUTPUTdir = inputs['OUTPUTdir']
numens = inputs['numens']
name_outputs = inputs['name_outputs']
filenames = inputs['filenames']
season = inputs['season']
area = inputs['area']
extreme = inputs['extreme']
timestep = inputs['timestep']

print('The name of the output files will be {0}'.format(name_outputs))
print('Number of ensemble members: {0}'.format(numens))

#____________Reading the netCDF file of 3Dfield, for all the ensemble members
var_ens = []
for ens in range(numens):
    ifile = filenames[ens]

    var, coords, aux_info = ecl.read_iris_nc(ifile)
    lat = coords['lat']
    lon = coords['lon']
    dates = coords['dates']
    time_units = aux_info['time_units']
    varunits = aux_info['var_units']
    #
    # #____________Selecting a season (DJF,DJFM,NDJFM,JJA)
    # var_season, dates_season = ecl.sel_season(var,dates,season)
    #
    # #____________Selecting only [latS-latN, lonW-lonE] box region
    # var_area, lat_area, lon_area = ecl.sel_area(lat,lon,var_season,area)

    #var_ens.append(var_area)
    var_ens.append(var)

inputs['var_units'] = varunits

print('Original var shape: (time x lat x lon)={0}'.format(var.shape))
# print('var shape after selecting season {0}: (time x lat x lon)={1}'.format(season,var_season.shape))
# print('var shape after selecting season {0} and area {1}: (time x lat x lon)={2}'.format(season,area,var_area.shape))
print('Check the number of ensemble members: {0}'.format(len(var_ens)))

if inputs['clim_compare']:
    if inputs['climat_file'] is None:
        raise ValueError('climat_file not specified. Either specify "climat_file" with the model climatology or set "clim_compare" to False.')
    clim, coords, aux_info = read_iris_nc(inputs['climat_file'])
    lat = coords['lat']
    lon = coords['lon']
    dates_clim = coords['dates']
else:
    clim = None
    dates_clim = None

if inputs['obs_compare']:
    if inputs['obs_file'] is None:
        raise ValueError('obs_file not specified')

    obs, coords, aux_info = read_iris_nc(inputs['obs_file'])
    lat = coords['lat']
    lon = coords['lon']
    dates_obs = coords['dates']
else:
    obs = None
    dates_obs = None

results = ecl.EnsClus(var_ens, lat, lon, dates, model_climatology = clim, model_climatology_dates = dates_clim, observed_anomaly = obs, observed_anomaly_dates = dates_obs, **inputs)

print('\n>>>>>>>>>>>> ENDED SUCCESSFULLY!! <<<<<<<<<<<<\n')

print('Check results in directory: {}\n'.format(OUTPUTdir))
print(ecl.datestamp()+'\n')

# os.system('mv {} {}'.format(logname, OUTPUTdir))
# os.system('cp {} {}'.format(file_input, OUTPUTdir))
