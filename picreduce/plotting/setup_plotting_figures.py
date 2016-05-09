
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12,}
params={'xtick.labelsize': 12,
        'ytick.labelsize': 12,}
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', **font)
plt.rcParams.update(params)

from matplotlib import style #requires matplotlib >1.4
style.use('grayscale')
import matplotlib.cm