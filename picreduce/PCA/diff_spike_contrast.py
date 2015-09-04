# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# functions necessary to calculate diffraction spike contrast (and return it as an radial average?)

# <codecell>

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm  # for log scaling of images, with automatic colorbar support
from matplotlib import gridspec
import gaussfitter
import astropy.io.fits as fits
import scipy.ndimage

def measure_spike(subregion,subregion_err,title_string="",ratio=20,ratio_err=10,rot_guess=90):    
    '''
    
    inputs:
        subregion: A approximately square slice of the science image around the diffraction spike of interest.
        subregion_err: the array 1sigma errors in the subregion
    
    Params:
        ratio: the ratio of the max of a bright PSF to the number of counts in the diffraction spike
        ratio_err: the 1sigma error of the ratio parameter
        rot_guess: initial guess for the rotation angle for the 2D gaussian. 
    Example:
        diffspik, diffspike_err=diff_spike_contrast.measure_spike(median[30:50,0:30],median_std[30:50,0:30],
                                                                    ratio=20,ratio_err=10,title_string=im_dset)

    '''
    fig=plt.figure(figsize=[16,4]) 
    gs1 = gridspec.GridSpec(1, 2)
    ax_list = [fig.add_subplot(ss) for ss in gs1]
    #subregion=image[ylo:yhi,xlo:xhi]
    #subregion_err=image[ylo:yhi,xlo:xhi]
    y,x=np.array(subregion.shape)/2.0
    
    ax_list[0]
    im1 = ax_list[0].imshow(subregion,origin='lower',interpolation='none')
    plt.colorbar(im1, ax=ax_list[0],use_gridspec=True) #http://stackoverflow.com/a/11558276/2142498
    ax_list[0].set_xlabel("cropped region and gaussian fit")
    params = (1,100,x,y,1,1,rot_guess)
    fitparameters, fittedgauss_image= gaussfitter.gaussfit(subregion,err=subregion_err,params=params,
                                                   fixed=[False, False, False, False, False, False, False],
                                                       rotate=True,returnfitimage=True,return_error=True)
    print(params)
    params,param_err= fitparameters
    (height, amplitude, x, y, width_x, width_y, rota) = params
    (height_err, amplitude_err, x_err, y_err, width_x_err, width_y_err, rota_err) =param_err
    ax_list[0].contour(fittedgauss_image,colors='yellow')
    
    residual=subregion-fittedgauss_image
    im2 = ax_list[1].imshow(residual,origin='lower',interpolation='none')
    plt.colorbar(im2,ax=ax_list[1],use_gridspec=True)
    ax_list[1].set_xlabel("residual")
    print(x,y)
    summation=np.sum(subregion-height)
    summation_err=np.sqrt(np.sum(subregion_err)**2)
    
    gauss_int=width_x*width_y*amplitude*2*np.pi
    gauss_int_err = gauss_int*np.sqrt((amplitude_err/amplitude)**2+(width_x_err/width_x)**2*(width_y_err/width_y)**2)
                                      
    plt.suptitle(title_string+ "\n Counts in the diffraction spike, \
    Gaussian:%.3g, summation:%.3g, percent diff: %.2g."\
    %( gauss_int, summation,100.*(gauss_int-summation)/gauss_int))
    
    # see nullertests/null_depth/diff_spikes/Find Bright PSF.ipynb
    brightMax=ratio*summation # see nullertests/null_depth/diff_spikes/Find Bright PSF.ipynb
    brightMax_err=np.sqrt(brightMax**2*((ratio_err/ratio)**2+(summation_err/summation)**2)) 
    brightMax_gauss=ratio*gauss_int # see nullertests/null_depth/diff_spikes/Find Bright PSF.ipynb
    brightMax_gauss_err=brightMax_gauss*np.sqrt((ratio_err/ratio)**2+(gauss_int_err/gauss_int)**2)
    
    #plt.subplot(143)
    #cen_y,cen_x=np.array(np.shape(subregion))/2.

    #oriented_subregion=np.roll(np.roll(subregion,np.int(np.round((cen_x-x))),axis=0),np.int(np.round((cen_y-y))),axis=1)
    #oriented_subregion=scipy.ndimage.interpolation.rotate(oriented_subregion,0,order=3,reshape=False)
    #raw=subregion/brightMax
    #im3=ax_list[3].imshow(raw,origin='lower',interpolation='none')
    #plt.colorbar(im3,ax=ax_list[2],use_gridspec=True)
    #ax_list[3].set_title("Diff. Spike Measured Contrast")

    #https://github.com/matplotlib/matplotlib/issues/829
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.90]) # adjust rect parameter while leaving some room for figtext.    
    return {"brightmax":brightMax,
            "brightmax_err":brightMax_err,
            "gaussianparams":params,
            "brightMax_gauss":brightMax_gauss,
            "brightMax_gauss_err":brightMax_gauss_err}

# <codecell>



# <codecell>

def get_nulled_frames(f,dset,null_state=34,n_skip=3):
    
    '''
    "diif_spike_contrast.get_nulled_frames is depreciated, use analysistools.slices.get_nulled_frames().
    example:
    #note! the first 3 values are skipped because they usually aren't really nulling

    '''
    print("diif_spike_contrast.get_nulled_frames is depreciated, use analysistools.slices.get_nulled_frames")
    nulled=np.where(f[dset][u'sci_header']['STATE'].flatten()==null_state)[0]
    good_sci_data=f[dset][u'sci'][:,:138,nulled[n_skip]:nulled[-1]]
    #null_diagnostic_plot(good_sci_data,oc,name=dset,xlo=45,xhi=93,ylo=45,yhi=93)#,grid=outer_grid)
    median=np.median(good_sci_data,axis=2)#-np.median(f[dset][u'sci'][0:10,0:10,nulled[n_skip]:nulled[-1]])
    std=np.std(good_sci_data,axis=2)

    #null_diagnostic_plot(good_sci_data,oc,name=dset,xlo=45,xhi=93,ylo=45,yhi=93)#,grid=outer_grid)
    
    '''good_sci_data= good_sci_data.swapaxes(0,2)
    good_sci_data = PynPoint._Util.mk_resizerecent(good_sci_data[:,:138,:],2,1)
    median=np.median(good_sci_data,axis=0)'''
            
    #median = recenter(median,(68, 68),boxsize=10)
    return good_sci_data,median,std

