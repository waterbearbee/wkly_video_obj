## Zhenheng Yang
## 07/16/2018
## ----------------------------------------------
## draw a heatmap around the distribution centers
## ----------------------------------------------

import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np

def heatmap(img, mu, sigma):
	
	im_size = img.shape
	heatmaps = np.zeros(im_size[:-1])
	mux, muy = mu[0], mu[1]
	sigmax, sigmay = sigma[0], sigma[1]
	x, y = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
	pos = np.empty(x.shape+(2,))
	pos[:,:,0] = x
	pos[:,:,1] = y
	rv = multivariate_normal([mux, muy], [[sigmax, 0], [0, sigmay]])
	wt_vals = rv.pdf(pos)
	wt_vals = wt_vals*(1.0/wt_vals.max())
	cmap = plt.cm.get_cmap('jet')
	img = cmap(wt_vals)
	plt.imshow(img)
	plt.savefig('1.png')