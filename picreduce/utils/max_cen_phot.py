import astropy.io.fits as fits
import poppy
import numpy as np
def max_cen_phot(in_array,radius_pixels=5,ref_width=0,fixed_center=None,verbose=True,**kwargs):
    '''
aperture photometry with auto centroiding.

        Parameters
        ----------
        in_array : array_like
            image to centroid and do photometry on
    radius_pixels: int
    defines photometry aperture
    fixed_center : tuple
    (y,x) coordinates of photometry aperture, defaults to None
    ref_width:  int
                pixels to add to the aperture radius for background measurement.
                defines background measuring photometry aperture beyond the primary radius,
                the average of which is subtracted.
    '''
    if fixed_center is None:
        cntry,cntrx=poppy.measure_centroid(fits.HDUList(fits.PrimaryHDU(in_array)),verbose=verbose,**kwargs)
    else:
        cntry,cntrx=fixed_center

    summed=0
    n_bins=0
    background=0
    n_bgnd=0
    maximum=0
    aperture_radius=radius_pixels
    for i in range(in_array.shape[0]-1):
        for j in range(in_array.shape[0]-1):
            r = np.sqrt((cntry-i)**2+(cntrx-j)**2)
            if r < aperture_radius:
                summed = summed + in_array[i,j]
                n_bins = n_bins+1.0
                if np.abs(in_array[i,j])>maximum:
                    maximum = in_array[i,j]
            if  (r > aperture_radius) & (r < aperture_radius+ref_width):
                background = background + in_array[i,j]
                n_bgnd = n_bgnd + 1.0
    if n_bgnd >0:
        mean_bgnd = background/n_bgnd
        corrected_sum=  summed - mean_bgnd*n_bins
    else:
         corrected_sum=  summed 
    if verbose:
        print("centroid: "+str(cntrx)+","+str(cntry))
        print("the total counts within r="+str(aperture_radius)+" of the centroid is:%.3e"%summed)
        print("the |maximum| counts within r="+str(aperture_radius)+" of the centroid is:%.3e"%maximum)

        if n_bgnd >0:
            print("the mean counts between r and r+r_width="+str(ref_width)+" of the centroid is:%.3e"%mean_bgnd)
            print("the reference subtracted counts within r="+str(aperture_radius)+" of the centroid is:%.3e"%corrected_sum)

    return corrected_sum
