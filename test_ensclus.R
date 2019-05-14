#test for multivariate eof
source("basis_functions.R")
#source("multieofs_function.R")
library(s2dverification)
library(reticulate)
library(stringr)

cart = "/data-hobbes/fabiano/temp/"

n_field = 25
filename = paste0(cart,"tas_2011_m11_ens",str_pad(1,6,pad='0'),".nc")
field = ncdf.opener.universal(filename, namevar = "tas")
field_arr = array(dim=c(n_field,dim(field$field)))
for (k in seq(1, n_field, 1)) {
  filename = paste0(cart,"tas_2011_m11_ens",str_pad(k,6,pad='0'),".nc")
  field = ncdf.opener.universal(filename, namevar = "tas")
  field_arr[k,,,] = field$field
}

lon = field$lon
lat = field$lat
#dates = as.character(field$time)
dates = field$time

print(dates)
source('ensclus_function.R')
out = ensclus(field_arr, lat, lon, dates, area = 'EAT', season = 'DJF') #, time)
