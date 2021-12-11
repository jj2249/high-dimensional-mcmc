import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

###--- Import spatial data ---###

### Read in the data
df = pd.read_csv('data.csv')

### Generate the arrays needed from the dataframe
data = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(data)
coords = [(xi[i],yi[i]) for i in range(N)]

### Subsample the original data set
subsample_factor = 3
idx = subsample(N, subsample_factor, seed=12)
G = get_G(N,idx)
c = G @ data


###--- MCMC ---####

### Set MCMC parameters
samps = 200
p1 = 0.001
p2 = 3.
# ls = np.linspace(p1, p2, samps)
ls = np.array([p1, 1.215, 10.])


# l = .462
n = 10000
beta = 0.2

# ### Set the likelihood and target, for sampling p(u|c)
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood

# for l in ls:
# # TODO: Complete Spatial Data questions (e), (f).
# 	K = GaussianKernel(coords, l)
# 	z = np.random.randn(N, )
# 	Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
# 	Kc_inverse = np.linalg.inv(Kc)
# 	K_inverse = Kc_inverse.T @ Kc_inverse
# 	u0 = Kc @ z


# 	upcn, accpcn = pcn(log_likelihood, u0, c, Kc, K_inverse, G, n, beta)

# 	print("Acceptance rate: "+str(accpcn))

# 	Cpred = predict_c(upcn)
# 	errorfield = np.abs(Cpred - data)
# 	print("length scale: "+str(l))
# 	print("Mean error: "+str(np.mean(errorfield)))

# 	## Plotting examples
# 	plot_2D(data, xi, yi, title='Bike Theft Data')                   # Plot bike theft count data
# 	plot_2D(c, xi[idx], yi[idx], title='Subsampled Data')      # Plot subsampled data
# 	plot_2D(Cpred, xi, yi, title='Predicted Data l={}'.format(l))      # Plot predicted data
# 	plot_2D(errorfield, xi, yi, title='Error Field l={}'.format(l))      # Plot error field
# plot_3D(data, xi, yi, title='Bike Theft Data')                   # Plot bike theft count data
# plot_3D(c, xi[idx], yi[idx], title='Subsampled Data')      # Plot subsampled data
# plot_3D(Cpred, xi, yi, title='Predicted Data')      # Plot predicted data
# plot_3D(errorfield, xi, yi, title='Error Field')      # Plot error field


### --- Heat map --- ###
samps = 200
p1 = 0.01
p2 = 10.
lvals = np.linspace(p1, p2, samps)
errors = []
for lval in tqdm(lvals):
	Kl = GaussianKernel(coords, lval)
	zl = np.random.randn(N, )
	Kcl = np.linalg.cholesky(Kl + 1e-6 * np.eye(N))
	Kcl_inverse = np.linalg.inv(Kcl)
	Kl_inverse = Kcl_inverse.T @ Kcl_inverse
	u0l = Kcl @ zl

	upcnl, accpcnl = pcn(log_likelihood, u0l, c, Kcl, Kl_inverse, G, n, beta)
	Cpredl = predict_c(upcnl)
	errorfieldl = np.abs(Cpredl - data)
	errors.append(np.mean(errorfieldl))

minidx = np.argmin(errors)

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(x=lvals, width=(p2-p1)/samps, height=errors, alpha=0.6, edgecolor='black', lw=0.5)
ax.set_xticks([0.01, lvals[minidx], lvals[-1]])
ax.set_xlabel('length scale')
ax.set_ylabel('Mean absolute error')
ax.set_ylim(bottom=1.)
fig.show()
print(lvals, errors)

plt.show()