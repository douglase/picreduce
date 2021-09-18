'''
fitting_funcs.py
part of the picreduce library
by Ewan Douglas in the course of analyzing PICTURE interferometer 
see: http://adsabs.harvard.edu/abs/2016PhDT.......100D

'''

from __future__ import division
import matplotlib.pyplot as plt
import astropy
import numpy as np
import scipy.ndimage
import logging

import wfe_funcs

import warnings
import numpy as np

import scipy.optimize

import sherpa.image

from sherpa import utils
from sherpa.models import Polynom2D,Polynom1D
from sherpa.data import Data2D,Data1D

from sherpa.optmethods import LevMar, NelderMead, MonCar
from sherpa.stats import Cash, LeastSq,Chi2ConstVar,Chi2
from sherpa.fit import Fit
from sherpa.estmethods import Confidence
from sherpa.estmethods import Covariance

def rms(array):
    return np.sqrt(np.mean(array**2))
def stdnew(array):
    return np.sqrt(np.sum(array-np.mean(array)))#/array.size

from astropy.modeling import models, fitting

def plane(m_x,m_y,c,x,y):
    
    return m_y*y + m_x*x + c

def fit_line_sherpa(x,y,y_error,verbose=True):
    '''
    https://github.com/DougBurke/sherpa-standalone-notebooks/blob/master/simulating%20a%202D%20image%20and%20a%20bit%20of%20error%20analysis.ipynb
    '''    
    model=Polynom1D('line')
    d = Data1D('data', x.flatten(), y.flatten(),staterror=y_error.flatten())
    model.c0.thaw()
    model.c1.thaw()
   
    fit = Fit(d, model, Chi2(), LevMar())
    result=fit.fit()
    meval = d.eval_model(model)
    fit.estmethod = Covariance()
    eres= fit.est_errors()

    if verbose: print(result)
    return result, eres, meval

def plane_func_1D(xy,c0,cx1,cy1):
    '''
    for use with `scipy.optimize.curve_fit`, which assumes `ydata = f(xdata, *params) + eps` 
    and names coefficients according the the Sherpa conventention

    x and y are expected to be mesh grids, e.g.
    y, x = np.mgrid[:z.shape[0], :z.shape[1]]

    '''
    x,y=xy
    plane= (cy1*y + cx1*x + c0)
    return plane
def fit_masked_plane_scipy(z,
                        error,
                        verbose=True,
                        optmethod="lm",
                        **kwargs):
    """
    meant to be drop in replacement for fit_plane_sherpa()
    assumes covariance matrix is diagonal and requires z be a masked array.
    """
    x,y = np.mgrid[:z.shape[0], :z.shape[1]]
    mask=z.mask
    
    scipyfit,scipycovar=scipy.optimize.curve_fit(plane_func_1D, 
                                   (x[mask != True].flatten(),y[mask != True].flatten()),
                                   1.0*z[mask != True],
                                        p0=(0,0,0),
                                   absolute_sigma=True,
                                   sigma=1.0*error[mask != True].flatten(),
                                   method=optmethod,
                                   **kwargs
                                   )
    
    err=np.sqrt(np.diagonal(scipycovar))

    if verbose:
        print("scipy fit constants",scipyfit)
        print("SNR",np.abs(scipyfit)/err)
    scipyplane=scipyfit[0] + x*scipyfit[1] + y*scipyfit[2]
    return scipyfit, err, scipyplane
    
def fit_plane_sherpa(z,
                         error,
                         statistic=Chi2,
                         verbose=True,
                         optmethod=LevMar):
    '''
    https://github.com/DougBurke/sherpa-standalone-notebooks/blob/master/simulating%20a%202D%20image%20and%20a%20bit%20of%20error%20analysis.ipynb
    '''
    y, x = np.mgrid[:z.shape[0], :z.shape[1]]
    
    model=Polynom2D('plane')
    d = Data2D('z', x.flatten(), y.flatten(), z.flatten(), shape=x.shape,staterror=error.flatten())
    d.ignore()
    d.notice(0)
    model.cy2.freeze()
    model.cx2.freeze()
    model.cy1.val=1
    model.cx1.val=-1
    model.cx1y2.freeze()
    model.cx2y1.freeze()
    model.cx1y1.freeze()
    model.cx2y2.freeze()
    fit = Fit(d, model, statistic(), optmethod())
    result=fit.fit()
    meval = d.eval_model(model).reshape(d.shape)
    fit.estmethod = Covariance()
    eres= fit.est_errors()

    if verbose: print(result)

    return result, eres, meval






def fit_plane(z,plotplane=False):
    '''
    http://docs.astropy.org/en/stable/modeling/#simple-2-d-model-fitting
    
    '''
    y, x = np.mgrid[:z.shape[0], :z.shape[1]]
    p_init = models.Polynomial2D(degree=1)
    fit_p = fitting.SLSQPLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)

    # Plot the data with the best-fit model
    if not plotplane:
        return p(x, y)
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)

    plt.imshow(z, origin='lower', interpolation='nearest')
    plt.title("Data")

    plt.subplot(1, 3, 2)
    plt.colorbar()


    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
    plt.title("Residual")
    plt.colorbar()
    return  p(x, y)



def chi_sq_stat(x,y,y_err):
    ''' pearson's chi squared statistic with known uncertainty-y_err
    Doesn't change any values,
    just calculates the chi squared of model x
    vs measurement y standard deviation of y_err.
    example:    
    '''
    result=np.sum((y-x)**2/y_err**2)
    dof=np.size(y)
    #print(result)
    return  {'chi^2':result,'dof':dof}



def zern_fit(coeffs,wf,data,weight=None,metric=np.std,N_coeff=25):
    '''
    generate a zernike basis from coeffs, subtract from data
    and measure the residual magnitude with metric.
    
     return the value of the function metric, 
     which defaults to standard deviation.
     alternatives could include rms() or np.mean() or median or...
     
    '''
    #re-import depencencies in the engine
    import wfe_funcs
    import numpy as np


    #im2shift=scipy.ndimage.shift(image2,(yx))
    #metric=np.sqrt(np.mean(im2shift-image1)**2) #alternate metric
    #print(coeffs)
    optic=wfe_funcs.zernike_optic(npix = wf.wavefront.shape[0],
                                  zern_coeffs =coeffs,nterms=N_coeff)
    sheared_composite = wfe_funcs.WFE_shear((optic.total_opd),
                                            -0.15,wf.pixelscale*1.3)
    diff=np.ma.masked_invalid(data - sheared_composite)
    val=metric(diff)
    #print(val)
    #plt.figure()
    #plt.imshow(diff)
    #plt.title(np.std(diff))
    #plt.colorbar()
    return val


def zern_fit_chi2(coeffs,wf,data,variance,N_coeff=25):
    '''
    generate a zernike basis from coeffs, subtract from data
    and measure the residual magnitude with metric.
    
     return the value of the function metric, 
     which defaults to standard deviation.
     alternatives could include rms() or np.mean() or median or...
     
    '''
    #re-import depencencies in the engine
    import wfe_funcs
    import numpy as np


    #im2shift=scipy.ndimage.shift(image2,(yx))
    #metric=np.sqrt(np.mean(im2shift-image1)**2) #alternate metric
    #print(coeffs)
    optic=wfe_funcs.zernike_optic(npix = wf.wavefront.shape[0],
                                  zern_coeffs =coeffs,nterms=N_coeff)
    sheared_composite = wfe_funcs.WFE_shear((optic.total_opd),
                                            -0.15,wf.pixelscale*1.3)
    diff = np.ma.masked_invalid(data - sheared_composite)
    val=np.sum((diff)**2/np.ma.masked_invalid(variance))
    #print(np.nanmax(variance))
    #print(val)
    #plt.figure()
    #plt.imshow(diff)
    #plt.title(np.std(diff))


    #plt.colorbar()
    return val

def crop_wfs(data_frame):
    return data_frame[7:-7,7:-7]
reload(wfe_funcs)
