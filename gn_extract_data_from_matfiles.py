execfile("core_functions.py")

import numpy as np
import os
from scipy.io import loadmat
from scipy.interpolate import griddata

#parameters
input_freqs = 120

input_time = 15
n_in = input_freqs*input_time


def extract_data(pathh):
	#extract all file names in the provided path
	a = os.listdir(pathh)
	nfiles = len(a)
	x = []
	y = []
	for f in a:
		d = loadmat(f)
		data = d['DATA']
		meta = d['META']
		#numb of samples in this birds data
		N = len(data)
		for i in np.arange(N):		
		    data = data[i][0]	
		    label = meta[i][0]
		    label = label['clustID'].astype('int')
			
			#resize the data by spline interpolation
			data = input_resize(data)
			
def input_resize(data,nypts,nxpts):
	grid_x,grid_y = np.mgrid([0:30:30j,0:120:50j])
	

