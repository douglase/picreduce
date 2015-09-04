'''
example code, shows nulling with the PICTURE (Planet Imaging Concept Testbed Using a Rocket Experiment Nuller.

'''
import sys,os
sys.path.insert(0, '../poppy')
home=os.path.expanduser('~')
import nulling_utils
import poppy
import null
import astropy
import numpy as np
import matplotlib.pylab as plt
import logging
_log = logging.getLogger('poppy')

global osys
def defosys(fov=20.2):
    
    plate_scale=0.158#"/px [sci]

    _log.setLevel(logging.DEBUG)
    osys = poppy.OpticalSystem(oversample=10)
    osys.addPupil( poppy.CircularAperture(radius=.25,pad_factor = 1.0))
    pixelscl = plate_scale
    
    osys.addDetector(pixelscale=pixelscl, fov_arcsec=fov)
    return osys


def nuller_dm():
    osys=defosys()
    meter_pixel_dm=340e-6/32.*42.55 #[wfs]

    dm=astropy.io.fits.open(home+'/projects/PICTURE/data/FITS/splinecongriddedcroppedrawdmdata.fits')
#convert DM surface error to WFE|
    dm_phase_array=dm[0].data*2.0
    lx,ly=dm_phase_array.shape
    pad_size=2048
    if lx >pad_size:
        print("padding problem")
    padded_dm = np.zeros([pad_size,pad_size])
    padded_dm[(pad_size-lx)/2:(pad_size-lx)/2+lx,(pad_size-lx)/2:(pad_size-lx)/2+lx]= dm_phase_array

    dm_phase=astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(padded_dm)])
    
    pupilmask_unscaled=poppy.FITSOpticalElement(transmission=home+'/projects/PICTURE/data/FITS/kenny_mask.fits',pixelscale=0.24/410.,oversample=10,opdunits='meters',rotation=-45)
    X=pupilmask_unscaled.getPhasor(osys.inputWavefront())
    plt.imshow(X.real)
    plt.colorbar()
    pupilmask=poppy.FITSOpticalElement(transmission=astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(data=np.int16(X))]),pixelscale=0.5/1024.,oversample=10,opdunits='meters',rotation=0)
    
    dm_flat_lowpass=astropy.io.fits.open(home+'/projects/PICTURE/data/FITS/boxcarred30pix.dm.fits')
    #convert DM surface error to WFE|
    dmflat_lowpass_phase_array=dm_flat_lowpass[0].data*2.0
    padded_dm_lowpass = np.zeros([pad_size,pad_size])
    padded_dm_lowpass[(pad_size-lx)/2:(pad_size-lx)/2+lx,(pad_size-lx)/2:(pad_size-lx)/2+lx]= dmflat_lowpass_phase_array
    
    DM_elem=            poppy.FITSOpticalElement(opd=dm_phase,pixelscale=meter_pixel_dm,
                                                oversample=10,opdunits='meters',rotation=-45-0.5)
    dmflat_lowpass_phase=astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(data=padded_dm_lowpass)])
    DMflat_lowpass_elem=poppy.FITSOpticalElement(opd=dmflat_lowpass_phase,pixelscale=meter_pixel_dm,
                                             oversample=10,opdunits='meters',rotation=-45-0.5)
    nuller_dm = null.NullingCoronagraph(osys,intensity_mismatch=.01, 
                                        display_intermediates=False,normalize='not', shear=0.15,
                                        phase_mismatch_fits=DM_elem,pupilmask=pupilmask,
                                        phase_mismatch_meters_pixel=meter_pixel_dm,
                                        phase_flat_fits=DMflat_lowpass_elem)
    return nuller_dm


def nuller_wfs():
    osys=defosys()
  
    nuller=null.NullingCoronagraph(osys,intensity_mismatch=0.00,
                               phase_mismatch_meters_pixel=170e-6*42.55,
                               display_intermediates=False,
                               normalize='not', shear=0.0)
    return nuller

