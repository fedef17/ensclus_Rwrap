#test for multivariate eof
source("basis_functions_mod.R")
source("ensclus_function.R")
library(s2dverification)
library(reticulate)
library(stringr)

#cart = "/data-hobbes/fabiano/temp/"
cart = "/home/fedefab/Scrivania/Research/Post-doc/data_temp/"

n_field = 25
filename = paste0(cart,"tas_2011_m11_ens",str_pad(1,6,pad='0'),".nc")#_15deg.nc")
field = ncdf.opener.universal(filename, namevar = "tas")
field_arr = array(dim=c(n_field,dim(field$field)))
for (k in seq(1, n_field, 1)) {
  filename = paste0(cart,"tas_2011_m11_ens",str_pad(k,6,pad='0'),".nc")#_15deg.nc")
  field = ncdf.opener.universal(filename, namevar = "tas")
  field_arr[k,,,] = field$field
}

lon = field$lon
lat = field$lat
#dates = as.character(field$time)
dates = field$time

gigi = apply(field_arr[,,,2:4],c(1,2,3),mean)

lat_lim = c(30, 87.5)
lon_lim = c(-80, 40)

print('Using numpcs')
resu1 = ensclus_Raw(gigi, lat, lon, numclus = 4, lon_lim = lon_lim, lat_lim = lat_lim, numpcs = 4)

print('Using perc explained')
resu2 = ensclus_Raw(gigi, lat, lon, numclus = 4, lon_lim = lon_lim, lat_lim = lat_lim, perc_explained = 60, flag_perc = TRUE)

print(resu1$closest_member)
print(resu2$closest_member)

# print(dates)
# out = ensclus(field_arr, lat, lon, dates, area = 'EAT', season = 'DJF', dir_OUTPUT = '/home/fedefab/Scrivania/Research/Post-doc/code_outputs/') #, time)
