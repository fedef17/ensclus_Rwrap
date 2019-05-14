#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.util as cutil
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats, optimize
import itertools as itt
import math

from sklearn.cluster import KMeans

from datetime import datetime
import pickle
from copy import deepcopy as dcopy

import iris
from cf_units import Unit

################################################

      ### Function library for ensclus ###

################################################
def datestamp():
    tempo = datetime.now().isoformat()
    tempo = tempo.split('.')[0]
    return tempo

def str_to_bool(s):
    if s == 'True' or s == 'T':
         return True
    elif s == 'False' or s == 'F':
         return False
    else:
         raise ValueError('Not a boolean value')

def std_outname(tag, inputs):
    name_outputs = '{}_{}_{}_{}clus'.format(tag, inputs['season'], inputs['area'], inputs['numclus'])

    if inputs['flag_perc']:
        name_outputs += '_{}perc'.format(inputs['perc'])
    else:
        name_outputs += '_{}pcs'.format(inputs['numpcs'])

    if inputs['obs_compare'] and inputs['fig_ref_to_obs']:
        name_outputs += '_refobs'
    else:
        name_outputs += '_refmod'

    return name_outputs


def read_inputs(nomefile, key_strings, n_lines = None, itype = None, defaults = None, verbose=False):
    """
    Standard reading for input files. Searches for the keys in the input file and assigns the value to variable. Returns a dictionary with all keys and variables assigned.

    :param key_strings: List of strings to be searched in the input file.
    :param itype: List of types for the variables to be read. Must be same length as key_strings.
    :param defaults: Dict with default values for the variables.
    :param n_lines: Dict. Number of lines to be read after key.
    """

    keys = ['['+key+']' for key in key_strings]

    if n_lines is None:
        n_lines = np.ones(len(keys))
        n_lines = dict(zip(key_strings,n_lines))
    elif len(n_lines.keys()) < len(key_strings):
        for key in key_strings:
            if not key in n_lines.keys():
                n_lines[key] = 1

    if itype is None:
        itype = len(keys)*[None]
        itype = dict(zip(key_strings,itype))
    elif len(itype.keys()) < len(key_strings):
        for key in key_strings:
            if not key in itype.keys():
                warnings.warn('No type set for {}. Setting string as default..'.format(key))
                itype[key] = str
        warnings.warn('Not all input types (str, int, float, ...) have been specified')

    if defaults is None:
        warnings.warn('No defaults are set. Setting None as default value.')
        defaults = len(key_strings)*[None]
        defaults = dict(zip(key_strings,defaults))
    elif len(defaults.keys()) < len(key_strings):
        for key in key_strings:
            if not key in defaults.keys():
                defaults[key] = None

    variables = []
    is_defaults = []
    with open(nomefile, 'r') as infile:
        lines = infile.readlines()
        # Skips commented lines:
        lines = [line for line in lines if not line.lstrip()[:1] == '#']

        for key, keystr in zip(keys, key_strings):
            deflt = defaults[keystr]
            nli = n_lines[keystr]
            typ = itype[keystr]
            is_key = np.array([key in line for line in lines])
            if np.sum(is_key) == 0:
                print('Key {} not found, setting default value {}\n'.format(key,deflt))
                variables.append(deflt)
                is_defaults.append(True)
            elif np.sum(is_key) > 1:
                raise KeyError('Key {} appears {} times, should appear only once.'.format(key,np.sum(is_key)))
            else:
                num_0 = np.argwhere(is_key)[0][0]
                try:
                    if typ == list:
                        cose = lines[num_0+1].rstrip().split(',')
                        coseok = [cos.strip() for cos in cose]
                        variables.append(coseok)
                    elif typ == dict:
                        # reads subsequent lines. Stops when finds empty line.
                        iko = 1
                        line = lines[num_0+iko]
                        nuvar = dict()
                        while len(line.strip()) > 0:
                            dictkey = line.split(':')[0].strip()
                            allvals = [cos.strip() for cos in line.split(':')[1].split(',')]
                            # if len(allvals) == 1:
                            #     allvals = allvals[0]
                            nuvar[dictkey] = allvals
                            iko += 1
                            line = lines[num_0+iko]
                        variables.append(nuvar)
                    elif nli == 1:
                        cose = lines[num_0+1].rstrip().split()
                        #print(key, cose)
                        if typ == bool: cose = [str_to_bool(lines[num_0+1].rstrip().split()[0])]
                        if typ == str: cose = [lines[num_0+1].rstrip()]
                        if len(cose) == 1:
                            if cose[0] is None:
                                variables.append(None)
                            else:
                                variables.append([typ(co) for co in cose][0])
                        else:
                            variables.append([typ(co) for co in cose])
                    else:
                        cose = []
                        for li in range(nli):
                            cos = lines[num_0+1+li].rstrip().split()
                            if typ == str: cos = [lines[num_0+1+li].rstrip()]
                            if len(cos) == 1:
                                if cos[0] is None:
                                    cose.append(None)
                                else:
                                    cose.append([typ(co) for co in cos][0])
                            else:
                                cose.append([typ(co) for co in cos])
                        variables.append(cose)
                    is_defaults.append(False)
                except Exception as problemha:
                    print('Unable to read value of key {}'.format(key))
                    raise problemha

    if verbose:
        for key, var, deflt in zip(keys,variables,is_defaults):
            print('----------------------------------------------\n')
            if deflt:
                print('Key: {} ---> Default Value: {}\n'.format(key,var))
            else:
                print('Key: {} ---> Value Read: {}\n'.format(key,var))

    return dict(zip(key_strings,variables))

###########################################################################
# READ/SAVE netcdf files

def check_increasing_latlon(var, lat, lon):
    """
    Checks that the latitude and longitude are in increasing order. Returns ordered arrays.

    Assumes that lat and lon are the second-last and last dimensions of the array var.
    """
    lat = np.array(lat)
    lon = np.array(lon)
    var = np.array(var)

    revlat = False
    revlon = False
    if lat[1] < lat[0]:
        revlat = True
        print('Latitude is in reverse order! Ordering..\n')
    if lon[1] < lon[0]:
        revlon = True
        print('Longitude is in reverse order! Ordering..\n')

    if revlat and not revlon:
        var = var[..., ::-1, :]
        lat = lat[::-1]
    elif revlon and not revlat:
        var = var[..., :, ::-1]
        lon = lon[::-1]
    elif revlat and revlon:
        var = var[..., ::-1, ::-1]
        lat = lat[::-1]
        lon = lon[::-1]

    return var, lat, lon


def regrid_cube(cube, ref_cube, regrid_scheme = 'linear'):
    """
    Regrids cube according to ref_cube grid. Default scheme is linear (cdo remapbil). Other scheme available: nearest and conservative (cdo remapcon).
    """
    if regrid_scheme == 'linear':
        schema = iris.analysis.Linear()
    elif regrid_scheme == 'conservative':
        schema = iris.analysis.AreaWeighted()
    elif regrid_scheme == 'nearest':
        schema = iris.analysis.Nearest()

    (nlat, nlon) = (len(cube.coord('latitude').points), len(cube.coord('longitude').points))
    (nlat_ref, nlon_ref) = (len(ref_cube.coord('latitude').points), len(ref_cube.coord('longitude').points))

    if nlat*nlon < nlat_ref*nlon_ref:
        raise ValueError('cube size {}x{} is smaller than the reference cube {}x{}!\n'.format(nlat, nlon, nlat_ref, nlon_ref))

    if np.all(cube.coord('latitude').points == ref_cube.coord('latitude').points) and np.all(cube.coord('longitude').points == ref_cube.coord('longitude').points):
        print('Grid check OK\n')
    else:
        nucube = cube.regrid(ref_cube, schema)

    return nucube


def transform_iris_cube(cube, regrid_to_reference = None, convert_units_to = None, extract_level_hPa = None, regrid_scheme = 'linear', adjust_nonstd_dates = True):
    """
    Transforms an iris cube in a variable and a set of coordinates.
    Optionally selects a level (given in hPa).
    TODO: cube regridding to a given lat/lon grid.

    < extract_level_hPa > : float. If set, only the corresponding level is extracted. Level units are converted to hPa before the selection.
    < force_level_units > : str. Sometimes level units are not set inside the netcdf file. Set units of levels to avoid errors in reading. To be used with caution, always check the level output to ensure that the units are correct.
    """
    print('INIZIO')
    ndim = cube.ndim
    datacoords = dict()
    aux_info = dict()
    ax_coord = dict()

    print(datetime.now())

    if regrid_to_reference is not None:
        cube = regrid_cube(cube, regrid_to_reference, regrid_scheme = regrid_scheme)

    print(datetime.now())

    if convert_units_to:
        if cube.units.name != convert_units_to:
            print('Converting data from {} to {}\n'.format(cube.units.name, convert_units_to))
            if cube.units.name == 'm**2 s**-2' and convert_units_to == 'm':
                cu = cu/9.80665
                cube.units = 'm'
            else:
                cube.convert_units(convert_units_to)

    print(datetime.now())
    data = cube.data
    print(datetime.now())
    aux_info['var_units'] = cube.units.name

    coord_names = [cord.name() for cord in cube.coords()]

    allco = ['lat', 'lon', 'level']
    allconames = dict()
    allconames['lat'] = np.array(['latitude', 'lat'])
    allconames['lon'] = np.array(['longitude', 'lon'])
    allconames['level'] = np.array(['level', 'lev', 'pressure', 'plev', 'plev8', 'air_pressure'])

    print(datetime.now())

    for i, nam in enumerate(coord_names):
        found = False
        if nam == 'time': continue
        for std_nam in allconames.keys():
            if nam in allconames[std_nam]:
                coor = cube.coord(nam)
                if std_nam == 'level':
                    coor.convert_units('hPa')
                datacoords[std_nam] = coor.points
                ax_coord[std_nam] = i
                found = True
        if not found:
            print('# WARNING: coordinate {} in cube not recognized.\n'.format(nam))

    print(datetime.now())
    if 'level' in datacoords.keys():
        if extract_level_hPa is not None:
            okind = datacoords['level'] == extract_level_hPa
            if np.any(okind):
                datacoords['level'] = datacoords['level'][okind]
                data = data.take(first(okind), axis = ax_coord['level'])
            else:
                raise ValueError('Level {} hPa not found among: '.format(extract_level_hPa)+(len(datacoords['level'])*'{}, ').format(*datacoords['level']))
        elif len(datacoords['level']) == 1:
            data = data.squeeze()

    if 'generic' in coord_names:
        data = data.squeeze()

    print(datetime.now())
    if 'time' in coord_names:
        time = cube.coord('time').points
        time_units = cube.coord('time').units
        dates = time_units.num2date(time) # this is a set of cftime._cftime.real_datetime objects
        time_cal = time_units.calendar

        if adjust_nonstd_dates:
            if dates[0].year < 1677 or dates[-1].year > 2256:
                print('WARNING!!! Dates outside pandas range: 1677-2256\n')
                dates = adjust_outofbound_dates(dates)

            if time_cal == '365_day' or time_cal == 'noleap':
                dates = adjust_noleap_dates(dates)
            elif time_cal == '360_day':
                dates = adjust_360day_dates(dates)

        datacoords['dates'] = dates
        aux_info['time_units'] = time_units.name
        aux_info['time_calendar'] = time_cal

    print(datetime.now())
    data, lat, lon = check_increasing_latlon(data, datacoords['lat'], datacoords['lon'])
    datacoords['lat'] = lat
    datacoords['lon'] = lon
    print('FINE')

    return data, datacoords, aux_info


def read_iris_nc(ifile, extract_level_hPa = None, select_var = None, regrid_to_reference = None, regrid_scheme = 'linear', convert_units_to = None, adjust_nonstd_dates = True, verbose = True, keep_only_maxdim_vars = True):
    """
    Read a netCDF file using the iris library.

    < extract_level_hPa > : float. If set, only the corresponding level is extracted. Level units are converted to hPa before the selection.
    < select_var > : str or list. For a multi variable file, only variable names corresponding to those listed in select_var are read. Redundant definition are treated safely: variable is extracted only one time.

    < keep_only_maxdim_vars > : keeps only variables with maximum size (excludes variables like time_bnds, lat_bnds, ..)
    """

    print('Reading {}\n'.format(ifile))

    fh = iris.load(ifile)

    cudimax = np.argmax([cu.ndim for cu in fh])
    ndim = np.max([cu.ndim for cu in fh])

    dimensions = [cord.name() for cord in fh[cudimax].coords()]

    if verbose: print('Dimensions: {}\n'.format(dimensions))

    if keep_only_maxdim_vars:
        fh = [cu for cu in fh if cu.ndim == ndim]

    variab_names = [cu.name() for cu in fh]
    if verbose: print('Variables: {}\n'.format(variab_names))
    nvars = len(variab_names)
    print('Field as {} dimensions and {} vars. All vars: {}'.format(ndim, nvars, variab_names))

    all_vars = dict()
    for cu in fh:
        all_vars[cu.name()] = transform_iris_cube(cu, regrid_to_reference = regrid_to_reference, convert_units_to = convert_units_to, extract_level_hPa = extract_level_hPa, regrid_scheme = regrid_scheme, adjust_nonstd_dates = adjust_nonstd_dates)

    if select_var is not None:
        print('Read variable: {}\n'.format(select_var))
        return all_vars[select_var]
    elif len(all_vars.keys()) == 1:
        print('Read variable: {}\n'.format(all_vars.keys()[0]))
        return all_vars.values()[0]
    else:
        print('Read all variables: {}\n'.format(all_vars.keys()))
        return all_vars


def adjust_noleap_dates(dates):
    """
    When the time_calendar is 365_day or noleap, nc.num2date() returns a cftime array which is not convertible to datetime (and to pandas DatetimeIndex). This fixes this problem, returning the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    # diffs = []
    for ci in dates:
        # coso = ci.isoformat()
        coso = ci.strftime()
        nudat = pd.Timestamp(coso).to_pydatetime()
        # print(coso, nudat)
        # if ci-nudat >= pd.Timedelta('1 days'):
        #     raise ValueError
        # diffs.append(ci-nudat)
        dates_ok.append(nudat)

    dates_ok = np.array(dates_ok)
    # print(diffs)

    return dates_ok


def adjust_outofbound_dates(dates):
    """
    Pandas datetime index is limited to 1677-2256.
    This temporary fix allows to handle with pandas outside that range, simply adding 1700 years to the dates.
    Still this will give problems with longer integrations... planned migration from pandas datetime to Datetime.datetime.
    """
    dates_ok = []
    diff = 2000

    for ci in dates:
        coso = ci.isoformat()
        listasp = coso.split('-')
        listasp[0] = '{:04d}'.format(int(listasp[0])+diff)
        coso = '-'.join(listasp)

        nudat = pd.Timestamp(coso).to_pydatetime()
        dates_ok.append(nudat)

    dates_ok = np.array(dates_ok)

    return dates_ok


def adjust_360day_dates(dates):
    """
    When the time_calendar is 360_day (please not!), nc.num2date() returns a cftime array which is not convertible to datetime (obviously)(and to pandas DatetimeIndex). This fixes this problem in a completely arbitrary way, missing one day each two months. Returns the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    strindata = '{:4d}-{:02d}-{:02d} 12:00:00'

    for ci in dates:
        firstday = strindata.format(ci.year, 1, 1)
        num = ci.dayofyr-1
        add_day = num/72 # salto un giorno ogni 72
        okday = pd.Timestamp(firstday)+pd.Timedelta('{} days'.format(num+add_day))
        dates_ok.append(okday.to_pydatetime())

    dates_ok = np.array(dates_ok)

    return dates_ok


def create_iris_cube(data, varname, varunits, iris_coords_list, long_name = None):
    """
    Creates an iris.cube.Cube instance.

    < iris_coords_list > : list of iris.coords.DimCoord objects (use routine create_iris_coord_list for standard coordinates).
    """
    # class iris.cube.Cube(data, standard_name=None, long_name=None, var_name=None, units=None, attributes=None, cell_methods=None, dim_coords_and_dims=None, aux_coords_and_dims=None, aux_factories=None)

    allcoords = []
    if not isinstance(iris_coords_list[0], iris.coords.DimCoord):
        raise ValueError('coords not in iris format')

    allcoords = [(cor, i) for i, cor in enumerate(iris_coords_list)]

    cube = iris.cube.Cube(data, standard_name = varname, units = varunits, dim_coords_and_dims = allcoords, long_name = long_name)

    return cube


def create_iris_coord_list(coords_points, coords_names, time_units = None, time_calendar = None, level_units = None):
    """
    Creates a list of coords in iris format for standard (lat, lon, time, level) coordinates.
    """

    coords_list = []
    for i, (cordpo, nam) in enumerate(zip(coords_points, coords_names)):
        cal = None
        circ = False
        if nam in ['latitude', 'longitude', 'lat', 'lon']:
            units = 'degrees'
            if 'lon' in nam: circ = True
        if nam == 'time':
            units = Unit(time_units, calendar = time_calendar)
        if nam in ['lev', 'level', 'plev']:
            units = level_units

        cord = create_iris_coord(cordpo, std_name = nam, units = units, circular = circ)
        coords_list.append(cord)

    return coords_list


def create_iris_coord(points, std_name, long_name = None, units = None, circular = False, calendar = None):
    """
    Creates an iris.coords.DimCoord instance.
    """
    # class iris.coords.DimCoord(points, standard_name=None, long_name=None, var_name=None, units='1', bounds=None, attributes=None, coord_system=None, circular=False)

    if std_name == 'longitude' or std_name == 'lon':
        circular = True
    if std_name == 'time' and (calendar is None or units is None):
        raise ValueError('No calendar/units given for time!')
        units = Unit(units, calendar = calendar)

    coord = iris.coords.DimCoord(points, standard_name = std_name, long_name = long_name, units = units, circular = circular)

    return coord


def save2Dncfield(lats,lons,variab,varname,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save2Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    lat = dataset.createDimension('lat', variab.shape[0])
    lon = dataset.createDimension('lon', variab.shape[1])

    # Create coordinate variables for 2-dimensions
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 2-d variable
    var = dataset.createVariable(varname, np.float64,('lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    #var.units = varunits

    lat[:]=lats
    lon[:]=lons
    var[:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 2D field [lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return


def save3Dncfield(lats, lons, variab, varname, varunits, dates, timeunits, time_cal, ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save3Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    time = dataset.createDimension('time', None)
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    time = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('time','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    time.units=timeunits
    time.calendar=time_cal
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    # Fill in times.
    time[:] = nc.date2num(dates, units = timeunits, calendar = time_cal)#, calendar = times.calendar)
    print(time_cal)
    print('time values (in units {0}): {1}'.format(timeunits,time[:]))
    print(dates)

    #print('time values (in units %s): ' % time)

    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 3D field [time x lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return


def save_N_2Dfields(lats,lons,variab,varname,varunits,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save a number N of 2D fields [latxlon]
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    num = dataset.createDimension('num', variab.shape[0])
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    num = dataset.createVariable('num', np.int32, ('num',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('num','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables.keys():
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    num[:]=np.arange(variab.shape[0])
    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The {0} 2D fields [num x lat x lon] are saved as \n{1}'.format(variab.shape[0], ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return

###### Selecting part of the dataset

def sel_area(lat,lon,var,area):
    '''
    GOAL
        Selecting the area of interest from a nc dataset.
    USAGE
        var_area, lat_area, lon_area = sel_area(lat,lon,var,area)

    :param area: str or list. If str: 'EAT', 'PNA', 'NH', 'Eu' or 'Med'. If list: a custom set can be defined. Order is (latS, latN, lonW, lonE).
    '''
    if area=='EAT':
        printarea='Euro-Atlantic'
        latN = 87.5
        latS = 30.0
        lonW =-80.0     #280
        lonE = 40.0     #40
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon

    elif area=='PNA':
        printarea='Pacific North American'
        latN = 87.5
        latS = 30.0
        lonW = 140.0
        lonE = 280.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If -180<lon<180, convert to 0<lon<360
        if lon.min() < 0:
            lon_new=lon+180
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon

    elif area=='NH':
        printarea='Northern Hemisphere'
        latN = 90.0
        latS = 0.0
        lonW = lon.min()
        lonE = lon.max()
        var_roll=var
        lon_new=lon

    elif area=='Eu':
        printarea='Europe'
        latN = 72.0
        latS = 27.0
        lonW = -22.0
        lonE = 45.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    elif area=='Med':
        printarea='Mediterranean'
        latN = 50.0
        latS = 25.0
        lonW = -10.0
        lonE = 40.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            print(var.shape)
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    elif (type(area) == list) or (type(area) == tuple) and len(area) == 4:
        lonW, lonE, latS, latN = area
        print('custom lat {}-{} lon {}-{}'.format(latS, latN, lonW, lonE))
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    else:
        raise ValueError('area {} not recognised'.format(area))

    latidx = (lat >= latS) & (lat <= latN)
    lonidx = (lon_new >= lonW) & (lon_new <= lonE)

    print('Area: ', lonW, lonE, latS, latN)
    # print(lat, lon_new)
    # print(latidx, lonidx)
    # print(var_roll.shape, len(latidx), len(lonidx))
    if var.ndim == 3:
        var_area = var_roll[:, latidx][..., lonidx]
    elif var.ndim == 2:
        var_area = var_roll[latidx, ...][..., lonidx]
    else:
        raise ValueError('Variable has {} dimensions, should have 2 or 3.'.format(var.ndim))

    return var_area, lat[latidx], lon_new[lonidx]


def sel_season(var, dates, season, cut = True):
    """
    Selects the desired seasons from the dataset.

    :param var: the variable matrix

    :param dates: the dates as extracted from the nc file.

    :param season: the season to be extracted.
    Formats accepted for season:
        - any sequence of at least 2 months with just the first month capital letter: JJA, ND, DJFM, ...
        - a single month with its short 3-letters name (First letter is capital): Jan, Feb, Mar, ...

    :param cut: bool. If True eliminates partial seasons.

    """

    dates_pdh = pd.to_datetime(dates)
    # day = pd.Timedelta('1 days')
    # dates_pdh_day[1]-dates_pdh_day[0] == day

    mesi_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_seq = 2*'JFMAMJJASOND'

    if season in month_seq and len(season) > 1:
        ind1 = month_seq.find(season)
        ind2 = ind1 + len(season)
        indxs = np.arange(ind1,ind2)
        indxs = indxs % 12 + 1

        mask = dates_pdh.month == indxs[0]
        for ind in indxs[1:]:
            mask = (mask) | (dates_pdh.month == ind)
    elif season in mesi_short:
        mask = (dates_pdh.month == mesi_short.index(season)+1)
    else:
        raise ValueError('season not understood, should be in DJF, JJA, ND,... format or the short 3 letters name of a month (Jan, Feb, ...)')

    var_season = var[mask, ...]
    dates_season = dates[mask]
    dates_season_pdh = pd.to_datetime(dates_season)

    #print(var_season.shape)

    if np.sum(mask) == 1:
        var_season = var_season[np.newaxis, :]

    #print(var_season.shape)

    if season in mesi_short or len(dates) <= 12:
        cut = False

    if cut:
        if (12 in indxs) and (1 in indxs):
            #REMOVING THE FIRST MONTHS (for the first year) because there is no previuos december
            start_cond = (dates_season_pdh.year == dates_pdh.year[0]) & (dates_season_pdh.month == indxs[0])
            if np.sum(start_cond):
                start = np.argmax(start_cond)
            else:
                start = 0

            #REMOVING THE LAST MONTHS (for the last year) because there is no following january
            end_cond = (dates_season_pdh.year == dates_pdh.year[-1]) & (dates_season_pdh.month == indxs[0])
            if np.sum(end_cond):
                end = np.argmax(end_cond)
            else:
                end = None

            var_season = var_season[start:end, ...]
            dates_season = dates_season[start:end]

    #print(var_season.shape)

    return var_season, dates_season


###########################################################################

def ens_anom(inputs, var_ens_tot, lat, lon, dates):
    '''
    \nGOAL: Computation of the ensemble anomalies based on the desired value from the input variable
    (it can be the percentile, mean, maximum, standard deviation or trend)
    OUTPUT: NetCDF files of ensemble mean of climatology, selected value and anomaly maps.
    '''

    OUTPUTdir = inputs['OUTPUTdir']
    numens = inputs['numens']
    name_outputs = inputs['name_outputs']
    season = inputs['season']
    area = inputs['area']
    extreme = inputs['extreme']
    varunits = inputs['var_units']

    print('The name of the output files will be {0}'.format(name_outputs))
    print('Number of ensemble members: {0}'.format(numens))

    # #____________Reading the netCDF file of 3Dfield, for all the ensemble members
    # var_ens = []
    # for ens in range(numens):
    #     ifile = filenames[ens]
    #
    #     var, coords, aux_info = read_iris_nc(ifile)
    #     lat = coords['lat']
    #     lon = coords['lon']
    #     dates = coords['dates']
    #     time_units = aux_info['time_units']
    #     varunits = aux_info['var_units']
    #
    #     #____________Selecting a season (DJF,DJFM,NDJFM,JJA)
    #     var_season, dates_season = sel_season(var,dates,season)
    #
    #     #____________Selecting only [latS-latN, lonW-lonE] box region
    #     var_area, lat_area, lon_area = sel_area(lat,lon,var_season,area)
    #
    #     var_ens.append(var_area)
    #
    # print('Original var shape: (time x lat x lon)={0}'.format(var.shape))
    # print('var shape after selecting season {0}: (time x lat x lon)={1}'.format(season,var_season.shape))
    # print('var shape after selecting season {0} and area {1}: (time x lat x lon)={2}'.format(season,area,var_area.shape))
    # print('Check the number of ensemble members: {0}'.format(len(var_ens)))
    var_ens = []
    for var in var_ens_tot:
        #____________Selecting a season (DJF,DJFM,NDJFM,JJA)
        if season is not None:
            var_season, dates_season = sel_season(var, dates, season)
        else:
            var_season = var
            dates_season = dates

        #____________Selecting only [latS-latN, lonW-lonE] box region
        var_area, lat_area, lon_area = sel_area(lat, lon, var_season, area)

        var_ens.append(var_area)

    if extreme=='mean':
        #Compute the time mean over the entire period, for each ensemble member # MEAN
        varextreme_ens=[np.mean(var_ens[i],axis=0) for i in range(numens)]

    elif len(extreme.split("_"))==2:
        #Compute the chosen percentile over the period, for each ensemble member # PERCENTILE
        q=int(extreme.partition("th")[0])
        varextreme_ens=[np.percentile(var_ens[i],q,axis=0) for i in range(numens)]

    elif extreme=='maximum':
        #____________Compute the maximum value over the period, for each ensemble member
        # MAXIMUM
        varextreme_ens=[np.max(var_ens[i],axis=0) for i in range(numens)]

    elif extreme=='std':
        #____________Compute the standard deviation over the period, for each ensemble member
        # STANDARD DEVIATION
        varextreme_ens=[np.std(var_ens[i],axis=0) for i in range(numens)]

    elif extreme=='trend':
        #____________Compute the linear trend over the period, for each ensemble member
        # TREND
        # Reshape grid to 2D (time, lat*lon)  -> Y
        #Y=[var_ens[i].reshape(var_ens[0].shape[0],var_ens[0].shape[1]*var_ens[0].shape[2])for i in range(numens)]
        #print('Reshaped (time, lat*lon) variable: ',Y[0].shape)
        trendmap=np.empty((var_ens[0].shape[1],var_ens[0].shape[2]))
        trendmap_ens=[]
        for i in range(numens):
            for la in range(var_ens[0].shape[1]):
                for lo in range(var_ens[0].shape[2]):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(range(var_ens[0].shape[0]),var_ens[i][:,la,lo])
                    trendmap[la,lo]=slope
            trendmap_ens.append(trendmap)
        varextreme_ens = trendmap_ens

    # print(len(varextreme_ens),varextreme_ens[0].shape)
    varextreme_ens_np = np.array(varextreme_ens)
    # print(varextreme_ens_np.shape)
    print('\n------------------------------------------------------------')
    print('Anomalies are computed with respect to the {0}'.format(extreme))
    print('------------------------------------------------------------\n')

    ensemble_mean = np.mean(varextreme_ens_np, axis = 0)
    #Compute and save the anomalies with respect to the ensemble
    ens_anomalies=varextreme_ens_np-np.mean(varextreme_ens_np,axis=0)
    #print(ofile)
    if inputs['netcdf_outputs']:
        varsave='ens_anomalies'
        ofile=os.path.join(OUTPUTdir,'ens_anomalies_{0}.nc'.format(name_outputs))
        print('Save the anomalies with respect to the ensemble:')
        print('ens_anomalies shape: (numens x lat x lon)={0}'.format(ens_anomalies.shape))
        save_N_2Dfields(lat_area,lon_area,ens_anomalies,varsave,varunits,ofile)

    #____________Compute and save the climatology
    vartimemean_ens=[np.mean(var_ens[i],axis=0) for i in range(numens)]
    ens_climatologies=np.array(vartimemean_ens)
    if inputs['netcdf_outputs']:
        varsave='ens_climatologies'
        ofile=os.path.join(OUTPUTdir,'ens_climatologies_{0}.nc'.format(name_outputs))
        #print(ofile)
        print('Save the climatology:')
        save_N_2Dfields(lat_area,lon_area,ens_climatologies,varsave,varunits,ofile)

    #____________Save the extreme
    ens_extreme=varextreme_ens_np
    if inputs['netcdf_outputs']:
        varsave='ens_extreme'
        ofile=os.path.join(OUTPUTdir,'ens_extreme_{0}.nc'.format(name_outputs))
        #print(ofile)
        print('Save the extreme:')
        save_N_2Dfields(lat_area,lon_area,ens_extreme,varsave,varunits,ofile)

    return ens_anomalies, ensemble_mean, dates, lat_area, lon_area, varunits


def clus_eval_indexes(PCs, centroids, labels):
    """
    Computes clustering evaluation indexes, as the Davies-Bouldin Index, the Dunn Index, the optimal variance ratio and the Silhouette value. Also computes cluster sigmas and distances.
    """
    ### Computing clustering evaluation Indexes
    numclus = len(centroids)
    inertia_i = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        inertia_i[i] = np.sum([np.sum((pcok-centroids[i])**2) for pcok in PCs[lab_clus]])

    clus_eval = dict()
    clus_eval['Indexes'] = dict()

    # Optimal ratio

    n_clus = np.empty(numclus)
    for i in range(numclus):
        n_clus[i] = np.sum(labels == i)

    mean_intra_clus_variance = np.sum(inertia_i)/len(labels)

    dist_couples = dict()
    coppie = list(itt.combinations(range(numclus), 2))
    for (i,j) in coppie:
        dist_couples[(i,j)] = LA.norm(centroids[i]-centroids[j])

    mean_inter_clus_variance = np.sum(np.array(dist_couples.values())**2)/len(coppie)

    clus_eval['Indexes']['Inter-Intra Variance ratio'] = mean_inter_clus_variance/mean_intra_clus_variance

    sigma_clusters = np.sqrt(inertia_i/n_clus)
    clus_eval['Indexes']['Inter-Intra Distance ratio'] = np.mean(dist_couples.values())/np.mean(sigma_clusters)

    # Davies-Bouldin Index
    R_couples = dict()
    for (i,j) in coppie:
        R_couples[(i,j)] = (sigma_clusters[i]+sigma_clusters[j])/dist_couples[(i,j)]

    DBI = 0.
    for i in range(numclus):
        coppie_i = [coup for coup in coppie if i in coup]
        Di = np.max([R_couples[cop] for cop in coppie_i])
        DBI += Di

    DBI /= numclus
    clus_eval['Indexes']['Davies-Bouldin'] = DBI

    # Dunn Index

    Delta_clus = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        distances = [LA.norm(pcok-centroids[i]) for pcok in PCs[lab_clus]]
        Delta_clus[i] = np.sum(distances)/n_clus[i]

    clus_eval['Indexes']['Dunn'] = np.min(dist_couples.values())/np.max(Delta_clus)

    clus_eval['Indexes']['Dunn 2'] = np.min(dist_couples.values())/np.max(sigma_clusters)

    # Silhouette
    sils = []
    for ind, el, lab in zip(range(len(PCs)), PCs, labels):
        lab_clus = labels == lab
        lab_clus[ind] = False
        ok_Pcs = PCs[lab_clus]
        a = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab]

        bs = []
        others = range(numclus)
        others.remove(lab)
        for lab_b in others:
            lab_clus = labels == lab_b
            ok_Pcs = PCs[lab_clus]
            b = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab_b]
            bs.append(b)

        b = np.min(bs)
        sils.append((b-a)/max([a,b]))

    sils = np.array(sils)
    sil_clus = []
    for i in range(numclus):
        lab_clus = labels == i
        popo = np.sum(sils[lab_clus])/n_clus[i]
        sil_clus.append(popo)

    siltot = np.sum(sil_clus)/numclus

    clus_eval['Indexes']['Silhouette'] = siltot
    clus_eval['clus_silhouettes'] = sil_clus

    clus_eval['Indexes']['Dunn2/DB'] = clus_eval['Indexes']['Dunn 2']/clus_eval['Indexes']['Davies-Bouldin']

    clus_eval['R couples'] = R_couples
    clus_eval['Inter cluster distances'] = dist_couples
    clus_eval['Sigma clusters'] = sigma_clusters

    return clus_eval


def eof_computation(var, lat, lon):
    """
    Compatibility version.

    Computes the EOFs of a given variable. In the first dimension there has to be different time or ensemble realizations of variable.

    The data are weighted with respect to the cosine of latitude.
    """
    # The data array is dimensioned (ntime, nlat, nlon) and in order for the latitude weights to be broadcastable to this shape, an extra length-1 dimension is added to the end:
    weights_array = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]

    start = datetime.now()
    solver = Eof(var, weights=weights_array)
    end = datetime.now()
    print('EOF computation took me {:7.2f} seconds'.format((end-start).total_seconds()))

    #ALL VARIANCE FRACTIONS
    varfrac = solver.varianceFraction()
    acc = np.cumsum(varfrac*100)

    #PCs unscaled  (case 0 of scaling)
    pcs_unscal0 = solver.pcs()
    #EOFs unscaled  (case 0 of scaling)
    eofs_unscal0 = solver.eofs()

    #PCs scaled  (case 1 of scaling)
    pcs_scal1 = solver.pcs(pcscaling=1)

    #EOFs scaled (case 2 of scaling)
    eofs_scal2 = solver.eofs(eofscaling=2)

    return solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac


def change_clus_order(centroids, labels, new_ord):
    """
    Changes order of cluster centroids and labels according to new_order.
    """
    numclus = int(np.max(labels)+1)

    labels_new = np.array(labels)
    for nu, i in zip(range(numclus), new_ord):
        labels_new[labels == i] = nu
    labels = labels_new

    centroids = centroids[new_ord, ...]

    return centroids, labels


def clus_order_by_frequency(centroids, labels):
    """
    Orders the clusters in decreasing frequency. Returns new labels and ordered centroids.
    """
    numclus = int(np.max(labels)+1)
    #print('yo',labels.shape)

    freq_clus = calc_clus_freq(labels)
    new_ord = freq_clus.argsort()[::-1]

    centroids, labels = change_clus_order(centroids, labels, new_ord)

    return centroids, labels


def calc_clus_freq(labels, numclus = None):
    """
    Calculates clusters frequency.
    """
    if numclus is None:
        numclus = int(np.max(labels)+1)
    #print('yo',labels.shape)

    num_mem = []
    for i in range(numclus):
        num_mem.append(np.sum(labels == i))
    num_mem = np.array(num_mem)

    freq_clus = 100.*num_mem/len(labels)

    return freq_clus


def Kmeans_clustering(PCs, numclus, order_by_frequency = True, n_init_sk = 600,  max_iter_sk = 1000):
    """
    Computes the Kmeans clustering on the given pcs.

    < param PCs > : the unscaled PCs of the EOF decomposition. The dimension should be already limited to the desired numpcs: PCs.shape = (numpoints, numpcs)
    < param numclus > : number of clusters.
    """

    start = datetime.now()
    clus = KMeans(n_clusters=numclus, n_init = n_init_sk, max_iter = max_iter_sk)

    clus.fit(PCs)
    centroids = clus.cluster_centers_
    labels = clus.labels_

    end = datetime.now()
    print('k-means algorithm took me {:7.2f} seconds'.format((end-start).total_seconds()))

    ## Ordering clusters for number of members
    centroids = np.array(centroids)
    labels = np.array(labels).astype(int)

    if order_by_frequency:
        centroids, labels = clus_order_by_frequency(centroids, labels)

    return centroids, labels


def ens_eof_kmeans(inputs, var_anom, lat, lon):
    '''
    Find the most representative ensemble member for each cluster.
    METHODS:
    - Empirical Orthogonal Function (EOF) analysis of the input file
    - K-means cluster analysis applied to the retained Principal Components (PCs)

    TODO:
    - Order clusters per frequency
    - Give the anomalies in input (not from file)

    '''

    # User-defined libraries

    OUTPUTdir = inputs['OUTPUTdir']
    numens = inputs['numens']
    name_outputs = inputs['name_outputs']
    numpcs = inputs['numpcs']
    perc = inputs['perc']
    numclus = inputs['numclus']

    # Either perc (cluster analysis is applied on a number of PCs such as they explain
    # 'perc' of total variance) or numpcs (number of PCs to retain) is set:
    if numpcs is not None and not inputs['flag_perc']:
        print('Considering fixed number of principal components: {0}'.format(numpcs))

    if perc is not None and inputs['flag_perc']:
        print('Considering percentage of explained variance: {0}%'.format(int(perc)))

    if (perc is None and numpcs is None) or (perc is None and inputs['flag_perc']) or (numpcs is None and not inputs['flag_perc']):
        raise ValueError('You have to specify either "perc" or "numpcs". Set "flag_perc" accordingly.')

    print('------------ EOF analysis -------------- \n')
    #----------------------------------------------------------------------------------------
    solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac = eof_computation(var_anom, lat, lon)

    acc=np.cumsum(varfrac*100)
    if inputs['flag_perc']:
        numpcs=min(enumerate(acc), key=lambda x: x[1]<=perc)[0]+1
        print('\nThe number of PCs that explain at least {}% of variance is {}'.format(perc,numpcs))
        exctperc=min(enumerate(acc), key=lambda x: x[1]<=perc)[1]
    if numpcs is not None:
        exctperc=acc[numpcs-1]
    if np.isnan(exctperc):
        print(acc)
        raise ValueError('NaN in evaluation of variance explained by first pcs')
    print('(the first {} PCs explain the {:5.2f}% of variance)'.format(numpcs,exctperc))


    #____________Compute k-means analysis using a subset of PCs
    print('__________________________________________________\n')
    print('k-means analysis')
    print('_____________________________________________\n')
    #----------------------------------------------------------------------------------------
    PCs = pcs_unscal0[:, :numpcs]

    centroids, labels = Kmeans_clustering(PCs, numclus)

    clus_eval = clus_eval_indexes(PCs, centroids, labels)
    for nam in clus_eval['Indexes'].keys():
        print(nam, clus_eval['Indexes'][nam])

    #____________Save labels
    if inputs['txt_outputs']:
        namef=os.path.join(OUTPUTdir,'labels_{0}.txt'.format(name_outputs))
        #np.savetxt(namef,labels,fmt='%d')
        filo = open(namef, 'w')
        #stringo = '{:6s} {:20s} {:8s}\n'.format('#', 'filename', 'cluster')
        stringo = '{:10s} {:10s}\n'.format('ens #', 'cluster')
        filo.write(stringo)
        filo.write(' \n')
        #for filnam, ii, lab in zip(inputs['filenames'], range(numens), labels):
        for ii, lab in zip(range(numens), labels):
            # indr = filnam.rindex('/')
            # filnam = filnam[indr+1:]
            # stringo = '{:6d} {:20s} {:8d}\n'.format(ii, filnam, lab)
            stringo = 'ens: {:6d} -> {:8d}\n'.format(ii, lab)
            filo.write(stringo)
        filo.close()

    #____________Compute cluster frequencies
    L=[]
    for nclus in range(numclus):
        cl=list(np.where(labels==nclus)[0])
        fr=len(cl)*100/len(labels)
        L.append([nclus,fr,cl])
    print('Cluster labels:')
    print([L[ncl][0] for ncl in range(numclus)])
    print('Cluster frequencies (%):')
    print([round(L[ncl][1],3) for ncl in range(numclus)])
    print('Cluster members:')
    print([L[ncl][2] for ncl in range(numclus)])

    #____________Find the most representative ensemble member for each cluster
    print('___________________________________________________________________________________')
    print('In order to find the most representative ensemble member for each cluster\n(which is the closest member to the cluster centroid)')
    print('the Euclidean distance between cluster centroids and each ensemble member is computed in the PC space')
    print('_______________________________________________________________________________')
    # 1)

    finalOUTPUT=[]
    repres=[]
    ens_mindist = []
    clus_std_dist = []
    clus_mean_dist = []

    for nclus in range(numclus):
        labok = (labels == nclus)
        norme = []
        for ens in np.arange(numens)[labok]:
            normens = centroids[nclus,:]-PCs[ens,:]
            norme.append(math.sqrt(sum(normens**2)))

        norme = np.array(norme)
        print('The distances between centroid of cluster {} and its members are:\n nums: {}\n dist: {}'.format(nclus, np.arange(numens)[labok], np.round(norme,3)))

        repr_clus = np.arange(numens)[labok][np.argmin(norme)]
        np.arange(numens)[labok]
        ens_mindist.append((repr_clus, norme.min()))
        repres.append(repr_clus)

        clus_mean_dist.append(np.mean(norme))
        clus_std_dist.append(np.std(norme))

        print('MINIMUM DISTANCE FOR CLUSTER {0} IS {1} --> member #{2}'.format(nclus, round(norme.min(),3), repr_clus))
        print('Mean INTRA-CLUSTER distance FOR CLUSTER {0} IS {1}\n'.format(nclus, clus_mean_dist[-1]))
        print('StdDev of INTRA-CLUSTER distance FOR CLUSTER {0} IS {1}\n'.format(nclus, clus_std_dist[-1]))

        txt='Closest ensemble member to centroid of cluster {0} is {1}\n'.format(nclus, repr_clus)
        finalOUTPUT.append(txt)

    if inputs['txt_outputs']:
        with open(OUTPUTdir+'RepresentativeEnsembleMembers_{0}.txt'.format(name_outputs), "w") as text_file:
            text_file.write(''.join(str(e) for e in finalOUTPUT))

        #____________Save the most representative ensemble members
        namef=os.path.join(OUTPUTdir,'repr_ens_{0}.txt'.format(name_outputs))
        filo = open(namef, 'w')
        filo.write('List of cluster representatives\n')
        #stringo = '{:10s} {:8s} -> {:20s}\n'.format('', '#', 'filename')
        stringo = '{:12s} {:10s}\n'.format('', '#')
        filo.write(stringo)
        filo.write(' \n')
        for ii in range(numclus):
            okin = repres[ii]
            # filnam = inputs['filenames'][okin]
            # indr = filnam.rindex('/')
            # filnam = filnam[indr+1:]
            #stringo = 'Cluster {:2d}: {:8d} -> {:20s}\n'.format(ii, okin, filnam)
            stringo = 'Cluster {:2d}: ens {:8d}\n'.format(ii, okin)
            filo.write(stringo)
        filo.close()

    return centroids, labels, ens_mindist, clus_eval

############################## PLOTS ##########################################
def color_brightness(color):
    return (color[0] * 299 + color[1] * 587 + color[2] * 114)/1000

def Rcorr(x,y):
    """
    Returns correlation coefficient between two array of arbitrary shape.
    """
    return np.corrcoef(x.flatten(), y.flatten())[1,0]


def E_rms(x,y):
    """
    Returns root mean square deviation: sqrt(1/N sum (xn-yn)**2).
    """
    n = x.size
    #E = np.sqrt(1.0/n * np.sum((x.flatten()-y.flatten())**2))
    E = 1/np.sqrt(n) * LA.norm(x-y)

    return E


def E_rms_cp(x,y):
    """
    Returns centered-pattern root mean square, e.g. first subtracts the mean to the two series and then computes E_rms.
    """
    x1 = x - x.mean()
    y1 = y - y.mean()

    E = E_rms(x1, y1)

    return E


def cosine(x,y):
    """
    Calculates the cosine of the angle between x and y. If x and y are 2D, the scalar product is taken using the np.vdot() function.
    """

    if x.ndim != y.ndim:
        raise ValueError('x and y have different dimension')
    elif x.shape != y.shape:
        raise ValueError('x and y have different shapes')

    if x.ndim == 1:
        return np.dot(x,y)/(LA.norm(x)*LA.norm(y))
    elif x.ndim == 2:
        return np.vdot(x,y)/(LA.norm(x)*LA.norm(y))
    else:
        raise ValueError('Too many dimensions')


def cosine_cp(x,y):
    """
    Before calculating the cosine, subtracts the mean to both x and y. This is exactly the same as calculating the correlation coefficient R.
    """

    x1 = x - x.mean()
    y1 = y - y.mean()

    return cosine(x1,y1)


def ens_plots(inputs, lat, lon, vartoplot, labels, ens_mindist, climatology = None, ensemble_mean = None, observation = None, varunits = ''):
    '''
    \nGOAL:
    Plot the chosen field for each ensemble
    NOTE:
    '''
    # User-defined libraries
    import matplotlib.path as mpath

    cmappa = cm.get_cmap(inputs['cmap'])
    cmappa_clus = cm.get_cmap(inputs['cmap_cluster'])

    # cmappa.set_under('violet')
    # cmappa.set_over('brown')

    OUTPUTdir = inputs['OUTPUTdir']
    numens = inputs['numens']
    name_outputs = inputs['name_outputs']
    numpcs = inputs['numpcs']
    perc = inputs['perc']
    numclus = inputs['numclus']
    varname = inputs['varname']

    plot_anomalies = inputs['plot_anomalies']

    n_color_levels = inputs['n_color_levels']
    n_levels = inputs['n_levels']
    draw_contour_lines = inputs['draw_contour_lines']

    tit=varname
    print('Number of clusters: {}'.format(numclus))

    if observation is not None:
        print(observation.shape)
        print('Plotting differences between the observation  and the model climatology\n')
        vartoplot3 = observation
    if climatology is not None:
        print(climatology.shape)
        print('Plotting differences with the model climatology instead that with the ensemble mean\n')
        vartoplot_new = []
        for var in vartoplot:
            vartoplot_new.append(var + ensemble_mean - climatology)
        vartoplot = np.array(vartoplot_new)

    # print(vartoplot2.shape)
    # print(vartoplot.shape)
    ofile = OUTPUTdir + 'Clusters_closest_ensmember_{}.nc'.format(name_outputs)
    print('Saving clustern anomalies (vs model climatology)\n')
    okins = [cos[0] for cos in ens_mindist]
    save_N_2Dfields(lat,lon,vartoplot[okins],'clus_anom_closer',varunits,ofile)

    if observation is not None and inputs['fig_ref_to_obs']:
        reference = vartoplot3
    else:
        reference = vartoplot

    mi = np.percentile(reference, 5)
    ma = np.percentile(reference, 95)
    oko = max(abs(mi), abs(ma))
    spi = 2*oko/(n_color_levels-1)
    spi_ok = np.ceil(spi*100)/100
    oko_ok = spi_ok*(n_color_levels-1)/2

    clevels = np.linspace(-oko_ok, oko_ok, n_color_levels)

    print('levels', len(clevels), min(clevels), max(clevels))

    colors = []
    valori = np.linspace(0.05,0.95,numclus)
    for cos in valori:
        colors.append(cmappa_clus(cos))

    for i, (col,val) in enumerate(zip(colors, valori)):
        #print(col, color_brightness(col))
        if color_brightness(col) > 0.6:
            #print('Looking for darker color')
            col2 = cmappa_clus(val+1.0/(3*numclus))
            col3 = cmappa_clus(val-1.0/(3*numclus))
            colori = [col, col2, col3]
            brighti = np.array([color_brightness(co) for co in colori]).argmin()
            colors[i] = colori[brighti]
    #colors = ['b','g','r','c','m','y','DarkOrange','grey']

    clat=lat.min()+abs(lat.max()-lat.min())/2
    clon=lon.min()+abs(lon.max()-lon.min())/2

    boundary = np.array([[lat.min(),lon.min()], [lat.max(),lon.min()], [lat.max(),lon.max()], [lat.min(),lon.max()]])
    bound = mpath.Path(boundary)

    proj = ccrs.PlateCarree()

    num_figs = int(np.ceil(1.0*numens/inputs['max_ens_in_fig']))
    numens_ok = int(np.ceil(numens/num_figs))

    side1 = int(np.ceil(np.sqrt(numens_ok)))
    side2 = int(np.ceil(numens_ok/float(side1)))

    for i in range(num_figs):
        fig = plt.figure(figsize=(24,14))
        for nens in range(numens_ok*i, numens_ok*(i+1)):
            nens_rel = nens - numens_ok*i
            #print('//////////ENSEMBLE MEMBER {}'.format(nens))
            ax = plt.subplot(side1, side2, nens_rel+1, projection=proj)
            ax.set_global()
            ax.coastlines()

            # use meshgrid to create 2D arrays
            xi,yi=np.meshgrid(lon,lat)

            # Plot Data
            if plot_anomalies:
                map_plot = ax.contourf(xi,yi,vartoplot[nens],clevels,cmap=cmappa, transform = proj, extend = 'both')
            else:
                map_plot = ax.contourf(xi,yi,vartoplot[nens],clevels, transform = proj, extend = 'both')

            latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
            ax.set_extent(latlonlim, crs = proj)

            # Add Title
            subtit = nens
            # title_obj=plt.title(subtit, fontsize=20, fontweight='bold', loc = 'left')
            title_obj = plt.text(-0.05, 1.05, subtit, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, fontweight='bold', zorder = 20)
            for nclus in range(numclus):
                if nens in np.where(labels==nclus)[0]:
                    okclus = nclus
                    bbox=dict(facecolor=colors[nclus], alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
                    title_obj.set_bbox(bbox)
                    #title_obj.set_backgroundcolor(colors[nclus])

            if nens == ens_mindist[okclus][0]:
                #rect = ax.patch
                rect = plt.Rectangle((-0.01,-0.01), 1.02, 1.02, fill = False, transform = ax.transAxes, clip_on = False, zorder = 10)#joinstyle='round')
                rect.set_edgecolor(colors[okclus])
                rect.set_linewidth(6.0)
                ax.add_artist(rect)
                #plt.draw()

        cax = plt.axes([0.1, 0.06, 0.8, 0.03]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.set_label(inputs['cb_label'], fontsize=20)

        plt.suptitle(tit+' ('+varunits+')', fontsize=35, fontweight='bold')

        plt.subplots_adjust(top=0.85)
        top    = 0.90  # the top of the subplots of the figure
        bottom = 0.13    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields

        namef = OUTPUTdir + '{}_{}_{}.{}'.format(varname, name_outputs, i, inputs['fig_format'])
        fig.savefig(namef)#bbox_inches='tight')

    print('____________________________________________________________________________________________')

    #plt.ion()
    fig2 = plt.figure(figsize=(18,12))
    print(numclus)
    side1 = int(np.ceil(np.sqrt(numclus)))
    side2 = int(np.ceil(numclus/float(side1)))
    print(side1,side2,numclus)

    for clu in range(numclus):
        ax = plt.subplot(side1, side2, clu+1, projection=proj)
        ok_ens = ens_mindist[clu][0]

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        map_plot = ax.contourf(xi,yi,vartoplot[ok_ens],clevels,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,vartoplot[ok_ens], n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)
        # proj_to_data = proj._as_mpl_transform(ax) - ax.transData
        # rect_in_target = proj_to_data.transform_path(bound)
        # ax.set_boundary(rect_in_target)

        title_obj = plt.title('Cluster {} - {:3.0f}% of cases'.format(clu, (100.0*sum(labels == clu))/numens), fontsize=24, fontweight='bold')
        title_obj.set_position([.5, 1.05])
        bbox=dict(facecolor=colors[clu], alpha = 0.5, edgecolor='black', boxstyle='round,pad=0.3')
        title_obj.set_bbox(bbox)
        #title_obj.set_backgroundcolor(colors[clu])

    cax = plt.axes([0.1, 0.07, 0.8, 0.03]) #horizontal
    cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=22)
    cb.set_label(inputs['cb_label'], fontsize=22)

    top    = 0.92  # the top of the subplots of the figure
    bottom = 0.13    # the bottom of the subplots of the figure
    left   = 0.02    # the left side of the subplots of the figure
    right  = 0.98  # the right side of the subplots of the figure
    hspace = 0.20   # the amount of height reserved for white space between subplots
    wspace = 0.05    # the amount of width reserved for blank space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # plot the selected fields
    namef=os.path.join(OUTPUTdir,'Clusters_{}_{}.{}'.format(varname,name_outputs, inputs['fig_format']))
    fig2.savefig(namef)#bbox_inches='tight')

################## OBSERVATIONSSSSSSSSSSSSSS

        ################# Observations vs climatology
    if observation is not None:
        fig4 = plt.figure(figsize=(8,6))
        ax = plt.subplot(projection=proj)

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        map_plot = ax.contourf(xi,yi,vartoplot3,clevels,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,vartoplot3, n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

        title_obj=plt.title('Observed anomaly', fontsize=20, fontweight='bold')
        title_obj.set_position([.5, 1.05])

        cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(inputs['cb_label'], fontsize=16)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.20    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Observed_anomaly_{}.{}'.format(name_outputs, inputs['fig_format']))
        fig4.savefig(namef)#bbox_inches='tight')

        fig4 = plt.figure(figsize=(8,6))
        ax = plt.subplot(projection=proj)

        ax.set_global()
        ax.coastlines(linewidth = 2)
        xi,yi=np.meshgrid(lon,lat)

        varensmean = ensemble_mean - climatology
        map_plot = ax.contourf(xi,yi,varensmean,clevels,cmap=cmappa, transform = proj, extend = 'both')
        if draw_contour_lines:
            map_plot_lines = ax.contour(xi,yi,varensmean, n_levels, colors = 'k', transform = proj, linewidth = 0.5)

        latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

        title_obj=plt.title('Ensemble mean', fontsize=20, fontweight='bold')
        title_obj.set_position([.5, 1.05])

        cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')#, labelsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(inputs['cb_label'], fontsize=16)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.20    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        # plot the selected fields
        namef=os.path.join(OUTPUTdir,'Ensemble_mean_{}.{}'.format(name_outputs, inputs['fig_format']))
        fig4.savefig(namef)#bbox_inches='tight')

        # Making the Taylor-like graph
        # Polar plot with
        bgcol = 'white'

        fig6 = plt.figure(figsize=(8,6))
        ax = fig6.add_subplot(111, polar = True)
        plt.title('Taylor diagram: predictions vs observed')
        ax.set_facecolor(bgcol)

        ax.set_thetamin(0)
        ax.set_thetamax(180)

        sigmas_pred = np.array([np.std(var) for var in vartoplot])
        sigma_obs = np.std(observation)
        corrs_pred = np.array([Rcorr(observation, var) for var in vartoplot])
        colors_all = [colors[clu] for clu in labels]
        angles = np.arccos(corrs_pred)
        print(corrs_pred.max(), corrs_pred.min())

        repr_ens = []
        for clu in range(numclus):
            repr_ens.append(ens_mindist[clu][0])

        ax.scatter([0.], [sigma_obs], color = 'black', s = 40, clip_on=False)

        only_numbers = False

        if not inputs['taylor_w_numbers']:
            ax.scatter(angles, sigmas_pred, s = 10, color = colors_all)
            ax.scatter(angles[repr_ens], sigmas_pred[repr_ens], color = colors, edgecolor = 'black', s = 40)
        else:
            #ax.scatter(angles[repr_ens], sigmas_pred[repr_ens], color = colors, edgecolor = 'black', s = 40, zorder = 20)
            if only_numbers:
                ax.scatter(angles, sigmas_pred, s = 0, color = colors_all)
                for i, (ang, sig, col) in enumerate(zip(angles, sigmas_pred, colors_all)):
                    zord = 5
                    siz = 3
                    if i in repr_ens:
                        zord = 21
                        siz = 4
                    gigi = ax.text(ang, sig, i, ha="center", va="center", color = col, fontsize = siz, zorder = zord, weight = 'bold')
            else:
                #ax.scatter(angles, sigmas_pred, s = 0, color = colors_all)
                #ax.scatter(angles[repr_ens], sigmas_pred[repr_ens], color = colors, edgecolor = col, s = 0)
                for i, (ang, sig, col) in enumerate(zip(angles, sigmas_pred, colors_all)):
                    zord = i + 1
                    siz = 4
                    if i in repr_ens:
                        zord = zord + numens
                        siz = 5
                        ax.scatter(ang, sig, color = col, alpha = 0.7, s = 60, zorder = zord, edgecolor = col)
                    else:
                        ax.scatter(ang, sig, s = 30, color = col, zorder = zord, alpha = 0.7)
                    gigi = ax.text(ang, sig, i, ha="center", va="center", color = 'white', fontsize = siz, zorder = zord, WEIGHT = 'bold')


        ok_cos = np.array([-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
        labgr = ['{:4.2f}'.format(co) for co in ok_cos]
        anggr = np.rad2deg(np.arccos(ok_cos))

        #ax.grid()
        plt.thetagrids(anggr, labels=labgr, frac = 1.1, zorder = 0)

        for sig in [1., 2., 3.]:
            circle = plt.Circle((sigma_obs, 0.), sig*sigma_obs, transform=ax.transData._b, fill = False, edgecolor = 'black', linestyle = '--')# color="red", alpha=0.1-0.03*sig)
            ax.add_artist(circle)

        top    = 0.88  # the top of the subplots of the figure
        bottom = 0.02    # the bottom of the subplots of the figure
        left   = 0.02    # the left side of the subplots of the figure
        right  = 0.98  # the right side of the subplots of the figure
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

        namef = OUTPUTdir + 'Taylor_diagram_{}.'.format(name_outputs) + inputs['fig_format']
        fig6.savefig(namef)

        fig7 = plt.figure(figsize=(8,6))
        ax = fig7.add_subplot(111)
        ax.set_facecolor(bgcol)

        biases = np.array([np.mean(var) for var in vartoplot])
        ctr_patt_RMS = np.array([E_rms_cp(var, observation) for var in vartoplot])
        RMS = np.array([E_rms(var, observation) for var in vartoplot])

        print('----------------------------\n')
        min_cprms = ctr_patt_RMS.argmin()
        print('The member with smallest centered-pattern RMS is member {} of cluster {}\n'.format(min_cprms, labels[min_cprms]))
        print('----------------------------\n')
        min_rms = RMS.argmin()
        print('The member with smallest absolute RMS is member {} of cluster {}\n'.format(min_rms, labels[min_rms]))
        print('----------------------------\n')
        min_bias = np.abs((biases - np.mean(observation))).argmin()
        print('The member with closest mean anomaly is member {} of cluster {}\n'.format(min_bias, labels[min_bias]))
        print('----------------------------\n')
        max_corr = corrs_pred.argmax()
        print('The member with largest correlation coefficient is member {} of cluster {}\n'.format(max_corr, labels[max_corr]))

        if not inputs['taylor_w_numbers']:
            ax.scatter(biases, ctr_patt_RMS, color = colors_all, s =10)
            ax.scatter(biases[repr_ens], ctr_patt_RMS[repr_ens], color = colors, edgecolor = 'black', s = 40)
        else:
            if only_numbers:
                ax.scatter(biases, ctr_patt_RMS, s = 0, color = colors_all)
                for i, (ang, sig, col) in enumerate(zip(biases, ctr_patt_RMS, colors_all)):
                    zord = 5
                    siz = 7
                    if i in repr_ens:
                        zord = 21
                        siz = 9
                    gigi = ax.text(ang, sig, i, ha="center", va="center", color = col, fontsize = siz, zorder = zord, weight = 'bold')
            else:
                for i, (ang, sig, col) in enumerate(zip(biases, ctr_patt_RMS, colors_all)):
                    zord = i + 1
                    siz = 7
                    if i in repr_ens:
                        zord = zord + numens
                        siz = 9
                        ax.scatter(ang, sig, color = col, alpha = 0.7, s = 200, zorder = zord, edgecolor = col)
                    else:
                        ax.scatter(ang, sig, s = 120, color = col, zorder = zord, alpha = 0.7)
                    gigi = ax.text(ang, sig, i, ha="center", va="center", color = 'white', fontsize = siz, zorder = zord, WEIGHT = 'bold')


        plt.xlabel('Mean anomaly (K)')
        plt.ylabel('Centered-pattern RMS difference [E\'] (K)')

        for sig in [1., 2., 3.]:
            circle = plt.Circle((np.mean(observation), 0.), sig*sigma_obs, fill = False, edgecolor = 'black', linestyle = '--')
            ax.add_artist(circle)

        plt.scatter(np.mean(observation), 0., color = 'black', s = 120, zorder = 5)
        plt.grid()

        namef = OUTPUTdir + 'Taylor_bias_vs_cpRMS_{}.'.format(name_outputs) + inputs['fig_format']
        fig7.savefig(namef)

        plt.close('all')

    return


def check_daily(dates):
    """
    Checks if the dataset is a daily dataset.
    """
    daydelta = pd.Timedelta('1 days')
    delta = dates[1]-dates[0]

    if delta == daydelta:
        return True
    else:
        return False

############################ MAIN FUNCTION ####################################
def EnsClus(var_ens, lat, lon, dates, numclus = 4, dir_OUTPUT = '.', exp_name = 'test', season = None, area = None, perc = 80, numpcs = 4, flag_perc = False, extreme = 'mean', model_climatology = None, model_climatology_dates = None, observed_anomaly = None, observed_anomaly_dates = None, varname = 'var', varunits = 'units', plot_outputs = True, netcdf_outputs = True, txt_outputs = True, **inputs):
    """
    EnsClus function.
    """

    lat = np.array(lat)
    lon = np.array(lon)
    print(lat.shape, lon.shape, var_ens.shape)
    if var_ens.shape[-1] != lon.shape[0]:
        okax = np.where(np.array(var_ens.shape) == lon.shape[0])[0][0]
        var_ens = np.swapaxes(var_ens, okax, -1)
    if var_ens.shape[-2] != lat.shape[0]:
        okax = np.where(np.array(var_ens.shape) == lat.shape[0])[0][0]
        var_ens = np.swapaxes(var_ens, okax, -2)
    print(lat.shape, lon.shape, var_ens.shape)

    print('Converting dates to Python datetime')
    print(dates[0], type(dates[0]))

    if type(dates[0]) is str:
        dates = pd.to_datetime(dates).to_pydatetime()
    elif type(dates[0]) in [float, np.float64, np.float32]:
        dates = np.array([datetime.utcfromtimestamp(da) for da in dates])

    print(dates[0], type(dates[0]))

    inputs['season'] = season
    inputs['area'] = area
    inputs['exp_name'] = exp_name
    inputs['dir_OUTPUT'] = dir_OUTPUT
    inputs['perc'] = perc
    inputs['numpcs'] = int(numpcs)
    inputs['flag_perc'] = flag_perc
    inputs['extreme'] = extreme
    inputs['var_units'] = varunits
    inputs['varname'] = varname
    inputs['numclus'] = int(numclus)
    inputs['plot_outputs'] = plot_outputs
    inputs['netcdf_outputs'] = netcdf_outputs
    inputs['txt_outputs'] = txt_outputs

    inputs['cb_label'] = varname + ' ({})'.format(varunits)

    keys = 'n_color_levels n_levels draw_contour_lines overwrite_output cmap cmap_cluster fig_format max_ens_in_fig check_best_numclus fig_ref_to_obs taylor_w_numbers plot_anomalies'
    keys = keys.split()

    defaults = dict()
    defaults['n_color_levels'] = 21
    defaults['n_levels'] = 5
    defaults['draw_contour_lines'] = False
    defaults['overwrite_output'] = True
    defaults['cmap'] = 'RdBu_r'
    defaults['cmap_cluster'] = 'nipy_spectral'
    defaults['fig_format'] = 'pdf'
    defaults['max_ens_in_fig'] = 30
    defaults['check_best_numclus'] = False
    defaults['fig_ref_to_obs'] = False
    defaults['taylor_w_numbers'] = True
    defaults['flag_perc'] = True
    defaults['plot_anomalies'] = True

    inputs['obs_compare'] = False
    if observed_anomaly is not None:
        inputs['obs_compare'] = True
        defaults['fig_ref_to_obs'] = True

    print(len(keys), len(defaults.keys()))

    for ke in keys:
        if ke in inputs.keys():
            if inputs[ke] is None:
                inputs[ke] = defaults[ke]
        else:
            inputs[ke] = defaults[ke]

    if dates is not None:
        dates_pdh = pd.to_datetime(dates)
        if check_daily(dates):
            inputs['timestep'] = 'day'
        else:
            inputs['timestep'] = 'month'

    print(inputs)

    OUTPUTdir = inputs['dir_OUTPUT'] + std_outname(inputs['exp_name'], inputs) + '/'

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

    inputs['numens'] = var_ens.shape[0]
    inputs['OUTPUTdir'] = OUTPUTdir

    #____________Building the name of output files
    inputs['name_outputs'] = std_outname(inputs['exp_name'], inputs)

    ####################### PRECOMPUTATION #######################################
    ens_anomalies, ensemble_mean, dates, lat, lon, varunits = ens_anom(inputs, var_ens, lat, lon, dates)

    #CHECK
    if model_climatology is not None:
        if model_climatology_dates is None:
            raise ValueError('model_climatology_dates not defined')
        if model_climatology.shape[-1] != len(lon):
            raise ValueError('Inconsistent number of longitudes in model_climatology: {}, should be  {}'.format(model_climatology.shape[-1], len(lon)))
        if model_climatology.shape[-2] != len(lat):
            raise ValueError('Inconsistent number of latitudes in model_climatology: {}, should be  {}'.format(model_climatology.shape[-2], len(lat)))

    if observed_anomaly is not None:
        if observed_anomaly_dates is None:
            raise ValueError('observed_anomaly_dates not defined')
        if observed_anomaly.shape[-1] != len(lon):
            raise ValueError('Inconsistent number of longitudes in observed_anomaly: {}, should be  {}'.format(observed_anomaly.shape[-1], len(lon)))
        if observed_anomaly.shape[-2] != len(lat):
            raise ValueError('Inconsistent number of latitudes in observed_anomaly: {}, should be  {}'.format(observed_anomaly.shape[-2], len(lat)))

    if model_climatology is not None:
        if season is not None:
            var_season, dates_season = sel_season(model_climatology, model_climatology_dates, season)
        else:
            var_season = var
            dates_season = model_climatology_dates

        climatology_tot, _, _ = sel_area(lat, lon, var_season, inputs['area'])
        climatology = np.mean(climatology_tot, axis = 0)
    else:
        climatology = None

    if observed_anomaly is not None:
        ### I need to extract the right year
        dates_obs_pdh = pd.to_datetime(observed_anomaly_dates)
        if inputs['timestep'] == 'month':
            delta = pd.Timedelta(weeks=1)
        elif inputs['timestep'] == 'day':
            delta = pd.Timedelta(hours=12)
        else:
            raise ValueError('timestep not understood')

        mask = (dates_obs_pdh > dates_pdh[0] - delta) & (dates_obs_pdh < dates_pdh[-1] + delta)
        obs = observed_anomaly[mask,:,:]
        dates_obs = observed_anomaly_dates[mask]

        if season is not None:
            var_season, dates_season = sel_season(obs, dates_obs, season)
        else:
            var_season = obs
            dates_season = dates_obs
        observation, _, _ = sel_area(lat, lon, var_season, inputs['area'])
        observation = np.mean(observation, axis = 0)
    else:
        observation = None


    ####################### EOF AND K-MEANS ANALYSES #############################
    if inputs['check_best_numclus']:
        print('Trying to determine best number of clusters..\n')
        indicators = []
        numclus_all = range(2,11)
        for numc in numclus_all:
            inputs['numclus'] = numc
            centroids, labels, ens_mindist, clus_eval = ens_eof_kmeans(inputs, ens_anomalies, lat, lon)
            indicators.append(clus_eval)

        if inputs['plot_outputs']:
            kiavi = clus_eval['Indexes'].keys()
            colors = []
            cmappa_clus = cm.get_cmap(inputs['cmap_cluster'])
            for cos in np.linspace(0.05,0.95,len(kiavi)):
                colors.append(cmappa_clus(cos))

            fig = plt.figure()
            for indx, col in zip(kiavi, colors):
                vals = [indica['Indexes'][indx] for indica in indicators]
                vals = np.array(vals)/np.max(vals)
                plt.plot(numclus_all, vals, label = None, color = col, linestyle = '--')
                plt.scatter(numclus_all, vals, label = indx, s = 20, color = col)

            plt.legend()
            plt.grid()
            plt.xlabel('Number of clusters')
            plt.ylabel('Normalized Indicator')
            fig.savefig(OUTPUTdir + 'Test_best_numclus_normalized.pdf')
            plt.close(fig)

            fig = plt.figure()
            for indx, col in zip(kiavi, colors):
                if 'Variance' in indx: continue
                vals = [indica['Indexes'][indx] for indica in indicators]
                plt.plot(numclus_all, vals, label = None, color = col, linestyle = '--')
                plt.scatter(numclus_all, vals, label = indx, s = 20, color = col)

            plt.legend()
            plt.grid()
            plt.xlabel('Number of clusters')
            plt.ylabel('Indicator')
            fig.savefig(OUTPUTdir + 'Test_best_numclus.pdf')
            plt.close(fig)

        results = dict()
        results['indicators'] = indicators
    else:
        centroids, labels, ens_mindist, clus_eval = ens_eof_kmeans(inputs, ens_anomalies, lat, lon)

        repr_ens = np.array([ens_mindist[nu][0] for nu in range(inputs['numclus'])])
        repr_fields = np.array([ens_anomalies[i]+ensemble_mean for i in repr_ens])

        results = dict()
        results['repr_ens_members'] = repr_ens
        results['member_clusters'] = np.array(labels)
        results['repr_fields'] = repr_fields

        if inputs['plot_outputs']:
            ens_plots(inputs, lat, lon, ens_anomalies, labels, ens_mindist, climatology = climatology, ensemble_mean = ensemble_mean, observation = observation, varunits = varunits)

    return results
