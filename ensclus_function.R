#source("basis_functions.R")
library(reticulate)

check.condaenv.ensclus <- function() {
	check = conda_list()
	trovato=FALSE
	for (nom in check[,1]) {
		ce = (nom == 'r-ensclus')
		if (ce) {trovato=TRUE}
	}

	listapac = c("netcdf4","cartopy","matplotlib","numpy","pandas","python=2.7","python-dateutil","scikit-learn","scipy","pickleshare=0.7.4=py27h09770e1_0")
	forgepack = c("eofs", "iris")

	if (trovato) {
		# controllo OK
		print('Python conda environment for ensclus already created!\n')
	} else {
		# creo l'environment
		conda_create('r-ensclus', packages = listapac)#, forge = TRUE)
		conda_install('r-ensclus', forgepack, forge = TRUE)
		conda_install('r-ensclus', 'matplotlib', forge = TRUE)
		print('Created Python conda environment for ensclus! You can now run ensclus function :)\n')
	}

	return(TRUE)
}

# EnsClus function
ensclus <- function(var_ens, lat, lon, dates, numclus = 4, dir_OUTPUT = './', exp_name = 'test', season = NULL, area = NULL, perc = 80, numpcs = 4, flag_perc = FALSE, extreme = 'mean', model_climatology = NULL, model_climatology_dates = NULL, observed_anomaly = NULL, observed_anomaly_dates = NULL, varname = 'var', varunits = 'units', plot_outputs = TRUE, netcdf_outputs = TRUE, txt_outputs = TRUE, ...) {
	check.condaenv.ensclus()
	use_condaenv('r-ensclus')

	# link zlib library inside conda environment to prevent reticulate in linking libraries of matplotlib
	check = conda_list()
	n <- length(check[,1])
	for (k in seq(1, n, 1)) {
		ce = (check[,1][k] == 'r-ensclus')
		if (ce) {env_path = check[,2][k]}
	}

	pos = regexpr('bin/python', env_path)
	okpath = substr(env_path, 1, pos-1)
	command = paste0('export LD_LIBRARY_PATH=', okpath, 'lib/:$LD_LIBRARY_PATH')
	print(command)
	system(command)

	source_python('ensclus_lib.py')
	print(datestamp())

	out = EnsClus(var_ens, lat, lon, dates, numclus = numclus, dir_OUTPUT = dir_OUTPUT, exp_name = exp_name, season = season, area = area, perc = perc, numpcs = numpcs, flag_perc = flag_perc, extreme = extreme, model_climatology = model_climatology, model_climatology_dates = model_climatology_dates, observed_anomaly = observed_anomaly, observed_anomaly_dates = observed_anomaly_dates, varname = varname, varunits = varunits, plot_outputs = plot_outputs, netcdf_outputs = netcdf_outputs, txt_outputs = txt_outputs, ...)

	print("Finalize...")
	#out=list(coeff=coefficient,variance=variance,eof_pattern=regression)

	return(out)
	#return(out)
}
