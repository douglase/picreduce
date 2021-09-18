
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.ndimage
import matplotlib
import logging
_log = logging.getLogger('poppy')
import astropy.io.fits as fits
from six import string_types



def radial_profile(HDUlist_or_filename=None, ext=0, EE=False, center=None, stddev=False, binsize=None, maxradius=None,minmax=True):
    """
    original code from: https://github.com/mperrin/poppy/utils.py

    Compute a radial profile of the image. 

    This computes a discrete radial profile evaluated on the provided binsize. For a version
    interpolated onto a continuous curve, see measure_radial().

    Code taken pretty much directly from pydatatut.pdf

    Parameters
    ----------
    HDUlist_or_filename : string
        what it sounds like.
    ext : int
        Extension in FITS file
    EE : bool
        Also return encircled energy (EE) curve in addition to radial profile?
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center. 
    binsize : float
        size of step for profile. Default is pixel size.
    stddev : bool
        Compute standard deviation in each radial bin, not average?
    minmax : bool
        return the mean, stdv, min and max in each radial bin?
    


    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """
    if isinstance(HDUlist_or_filename, string_types):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data
    pixelscale = HDUlist[ext].header['PIXELSCL']


    if maxradius is not None:
        raise NotImplemented("add max radius")


    if binsize is None:
        binsize=pixelscale

    y,x = np.indices(image.shape)
    if center is None:
        # get exact center of image
        #center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple( (a-1)/2.0 for a in image.shape[::-1])

    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2) *pixelscale / binsize # radius in bin size steps
    ind = np.argsort(r.flat)


    sr = r.flat[ind]
    sim = image.flat[ind]
    ri = sr.astype(int)
    deltar = ri[1:]-ri[:-1] # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=float) # cumulative sum to figure out sums for each bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile=tbin/nr

    #pre-pend the initial element that the above code misses.
    radialprofile2 = np.empty(len(radialprofile)+1)
    if rind[0] != 0:
        radialprofile2[0] =  csim[rind[0]] / (rind[0]+1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]                       # otherwise if there's just one then just take it. 
    radialprofile2[1:] = radialprofile
    rr = np.arange(len(radialprofile2))*binsize + binsize*0.5  # these should be centered in the bins, so add a half.

    if (stddev) or (minmax):
        stddevs = np.zeros_like(radialprofile2)
        mins = np.zeros_like(radialprofile2)
        maxes = np.zeros_like(radialprofile2)

        r_pix = r * binsize
        for i, radius in enumerate(rr):
            if i == 0: wg = np.where(r < radius+ binsize/2)
            else: 
                wg = np.where( (r_pix >= (radius-binsize/2)) &  (r_pix < (radius+binsize/2)))
                #wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
            stddevs[i] = image[wg].std()
            if image[wg].size != 0:
                mins[i]=np.min(image[wg])
                maxes[i]=image[wg].max()
        
        if (minmax) and (stddev):
            return (rr, radialprofile2,stddevs,mins,maxes)
        elif minmax:
            return (rr, mins,maxes)
        elif stddev:
            return (rr, stddevs)

    if not EE:
        return (rr, radialprofile2)
    else:
        #weighted_profile = radialprofile2*2*np.pi*(rr/rr[1])
        #EE = np.cumsum(weighted_profile)
        EE = csim[rind]
        return (rr, radialprofile2, EE) 
