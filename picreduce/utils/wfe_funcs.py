import poppy
import numpy as np
import null
import os,sys
import scipy.ndimage
#home= os.path.expanduser("~")
#sys.path.insert(0, home+'/projects/PICTURE/code/picreduce/picreduce/utils/')

def WFE_shear(pupil,shear,pixelscale):
    arm1 = pupil
    arm2 = null.sheararray(pupil,shear,pixelscale)
    output = null.sheararray(arm1 - arm2,-shear/2.,pixelscale)
    return output

class zernike_model():
    def __init__(self, npix=512,**kwargs):
        self.zernike = poppy.zernike.zernike_basis(nterms=nterms, npix=npix,**kwargs)
        self.z_coeff = np.ones(self.zernike.shape[0])
        
class zernike_optic(poppy.optics.AnalyticOpticalElement):
    """
    """

    def __init__(self, zern_coeffs=None, 
                 planetype=poppy.PlaneType.intermediate,
                 name='Zernike Optical Element',
                 **kwargs):
        poppy.AnalyticOpticalElement.__init__(self, name=name, planetype=planetype)

        self.zernike_basis = poppy.zernike.zernike_basis(**kwargs)
        if zern_coeffs == None:
            self.zern_coeffs = np.ones(self.zernike_basis.shape[0])
        else:
            self.zern_coeffs = zern_coeffs
    @property
    def total_opd(self):
        self.phase = self.zern_coeffs*self.zernike_basis.T
        self.phase = np.sum(self.phase,axis=2) #sum all the Zernikes
        return self.phase
    def getPhasor(self, wave):
        """ return complex phasor for the zernike basis

    
        """
        
        return np.exp(self.total_opd*1j)


def V(A,B,C,D):
    '''
    Calculation of visibility from 4 frames, assuming successive 90 degree shifts between each frame.
    '''
    vis = 2.0*np.sqrt(((A-C)**2+(B-D)**2)/(A+B+C+D)**2)
    return vis
def m_vis(A,B,C,D):
    '''
    simple calculation of visibility from 4 frames, with no assumptions about how the measurements are made
    '''
    vis = (np.max([A,B,C,D],axis=0)-np.min([A,B,C,D],axis=0))/(np.max([A,B,C,D],axis=0)+np.min([A,B,C,D],axis=0))
    return vis
def I(A,B,C,D):
    return (A+B+C+D)

def phase(A,B,C,D):
    '''
    '''
    phase = np.arctan2((A-C),(B-D))
    return phase

def phase3step(A,B,C,D):
    '''
    a three step algorithm
    for example section 4.1 of
    http://fp.optics.arizona.edu/jcwyant/Optics513/ChapterNotes/Chapter05/Notes/Phase%20Shifting%20Interferometry.nb.pdf

    i1-->B,i2-->C,i3-->D
    '''
    print('ignoring A')
    phase = np.arctan2((B - D),(-B + 2.0*C - D))
    return phase

def phase_variance3step_Corrected(A,B,C,D):
    '''
assuming $\sigma_B=\sqrt{B}$
    '''
    variance =  (B*(C-D)**2+\
            C*(D-B)**2+\
            D*(B-C)**2)/(B**2-2*B*C+2*C**2-2*C*D+D**2)**2 
    return variance


def phase_variance3step(A,B,C,D):
    '''
assuming $\sigma_B=\sqrt{B}$

$\sigma(\phi_{BAC})^2 = \sigma_B^2(\frac{\partial Q}{\partial B})^2...=
\sigma_b^2\frac{(c-d)}{(b^2-2bc+2c^2-2cd+d^2)}^2
+\frac{(d-b)}{(b^2-2bc+2c^2-2cd+d^2))}^2\sigma_c^2+
\sigma_D^2(\frac{b-c}{b^2-2bc+2c^2-2cd+d^2})^2$

$\sigma(\phi_{BAC})^2 = \sigma_B^2(\frac{\partial Q}{\partial B})^2...=
(B(C-D)^2+D(B-C)+C(D-B)^2)*(\frac{1}{B^2-2BC+2C^2-2CD+D^2})^2$


    '''
    variance =  (B*(C-D)**2+\
            C*(D-B)**2+\
            D*(B-C)**2)/(B**2-2*B*C+2*C**2-2*C*D+D**2)**2 
    return variance

def phase_variance(A,B,C,D):
    '''
    $\sigma^2 = (\frac{(B-D)}{(B-D)^2 + (A-C)^2)})^2(\sigma_A^2+\sigma_C^2) + 
    (\frac{(A-C)}{((B-D)^2 + (A-C)^2)})^2(\sigma_B^2+\sigma_D^2)$

    assuming $\sigma_B=\sqrt{B}$ (Therefore calculations need to be in electrons/second)



    
    this is given in Wyant 1975, Applied Optics. Eq. 22, 
    derived from Worthing and Geffner 1943.


    Mendillo 2013 (dissertation) p. 190 eq. 5.20 derives the same expression, 
    except that expression averages the phase measurement across the number of 
    WFS pixels per DM actuator. This doesn't apply when we are just trying to measure 
    how well the
    WFS works, it does matter when you are characterizing the WCS.
    '''
    variance = ((B-D)/((B-D)**2 + (A-C)**2))**2*(A+C) + ((A-C)/((B-D)**2 + (A-C)**2))**2*(B+D)
    return variance
    

def crop_rot_wfs(data_frame):
    return scipy.ndimage.rotate(data_frame,180-45)[17:-15,17:-15]

def crop_rot_raw(data_frame):
    return scipy.ndimage.rotate(data_frame,180-45)[7:-7,7:-7]