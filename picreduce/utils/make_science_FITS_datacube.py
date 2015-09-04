# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import scipy.io
import numpy as np
import astropy.io.fits as fits
directory='/Users/edouglas/projects/PICTURE/data/flight_sequence_20140530/flight_sequence_20140530/gsedata.idl.05302014.58866/'

# <codecell>

phasefiles=glob.glob(directory+'/*.u.idl')
sci_list=[]
for i,unwrappedfile in enumerate(phasefiles):
    try:
        unwrappedapped=scipy.io.readsav(unwrappedfile)
        max_error=unwrappedapped['data'].max()*675.0/(2*np.pi)
        #uncomment this conditional if you want to filter out for only certain phase error regimes:
        #if (max_error>650) & (max_error !=0) & (max_error<100) & (max_error !=0):
        #    continue

        #check if there are other frames with this timestamp
        if unwrappedfile[:-21] == phasefiles[i-1][:-21]:
            continue            
        scienceframe_s=glob.glob(unwrappedfile[:-21]+'*.sci.s.idl')
        for frame in scienceframe_s:
            #print(frame)
            sci_frame=scipy.io.readsav(frame)
            #fits.writeto(frame+".fits",sci_frame['data'])
            sci_list.append(sci_frame['data'])
    except Exception,err:    
        print(err,unwrappedfile,frame)

try:
    fits.writeto(directory+"/sci_cube.fits",np.array(sci_list))
except Exception, err:
    print(err)

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


