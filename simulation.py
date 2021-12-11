import numpy as np
import matplotlib.pyplot as plt
from functions import *


###--- Data Generation ---###

### Inference grid defining {ui}i=1,Dx*Dy
Dx = 16
Dy = 16
N = Dx * Dy     # Total number of coordinates
points = [(x, y) for y in np.arange(Dx) for x in np.arange(Dy)]                # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])    # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists



### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 1
idx = subsample(N, subsample_factor)
M = len(idx)                                                                   # Total number of data points

### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
l = 0.3
K = GaussianKernel(coords, l)
z = np.random.randn(N, )
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
Kc_inverse = np.linalg.inv(Kc)
K_inverse = Kc_inverse.T @ Kc_inverse
u = Kc @ z

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u# + np.random.randn(M)


###--- MCMC ---####

### Set MCMC parameters
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

### Sample from prior for MCMC initialisation
u0 = Kc @ np.random.normal(size=N)
### Chains for q1b
Xgrw, accgrw = grw(log_target, u0, v, Kc, K_inverse, G, n, beta)
Xpcn, accpcn = pcn(log_likelihood, u0, v, Kc, K_inverse, G, n, beta)
Xgrw = np.array(Xgrw)
Xpcn = np.array(Xpcn)
### Mean inferred fields
ugrw = np.mean(Xgrw, axis=0)
upcn = np.mean(Xpcn, axis=0)
ugrwbi = np.mean(Xgrw[1000:], axis=0)
upcnbi = np.mean(Xpcn[1000:], axis=0)
# TODO: Complete Simulation questions (a), (b).

### Find error fields
errorgrw = np.abs(ugrw-u)
errorpcn = np.abs(upcn-u)
errorgrwbi = np.abs(ugrwbi-u)
errorpcnbi = np.abs(upcnbi-u)
print("GRW wo bi: "+str(np.mean(errorgrw)))
print("pCN wo bi: "+str(np.mean(errorpcn)))
print("GRW w bi: "+str(np.mean(errorgrwbi)))
print("pCN w bi: "+str(np.mean(errorpcnbi)))
### Plotting examples
fig1 = plot_3D(u, x, y)                                      # Plot original u surface
fig2 = plot_result(u, v, x, y, x[idx], y[idx])               # Plot original u with data v
fig1.show()
fig2.show()

### GRW field
fig3 = plot_3D(ugrw, x, y)
fig3.show()
fig5 = plot_result(ugrw, v, x, y, x[idx], y[idx])
fig5.show()

### pCN field
fig4 = plot_3D(upcn, x, y)
fig4.show()
fig6 = plot_result(upcn, v, x, y, x[idx], y[idx])
fig6.show()

### Error fields
fig7 = plot_3D(errorgrw, x, y)
fig7.show()
fig8 = plot_3D(errorpcn, x, y)
fig8.show()

### Acceptance probs
print("grw acceptance: " + str(accgrw))
print("pcn acceptance: " + str(accpcn))


###--- Probit transform ---###
t = probit(v)       # Probit transform of data

# TODO: Complete Simulation questions (c), (d).
# Tpcn, accTpcn = pcn(log_probit_likelihood, u0, t, Kc, K_inverse, G, n, beta)
# Tinf = predict_t(Tpcn)
# Tthres = threshold(Tinf)

# print("Probit pCN acceptance: " + str(accTpcn))
# prediction_error = np.mean(np.abs(Tthres-probit(u)))
# print("Hard assignment prediction error: "+str(prediction_error))


## length scale heat map
# grid_samps = 50
# lvals = np.linspace(0.01, 10, grid_samps)
# grid_errors = []
# accrates = []
# for lval in tqdm(lvals):
# 	Kl  = GaussianKernel(coords, lval)
# 	Kcl = np.linalg.cholesky(Kl + 1e-6 * np.eye(N))
# 	Kcl_inverse = np.linalg.inv(Kcl)
# 	Kl_inverse = Kcl_inverse.T @ Kcl_inverse
# 	u0l = Kcl @ np.random.normal(size=N)
# 	chain, accrate = pcn(log_probit_likelihood, u0l, t, Kcl, Kl_inverse, G, n, beta)
# 	threshvals = threshold(predict_t(chain))
# 	error = np.mean(np.abs(threshvals-probit(u)))
# 	grid_errors.append(error)
# 	accrates.append(accrate)

# minidx = np.argmin(np.array(grid_errors))
# print(grid_errors)

### Plotting examples
### Original data and subsampled data
# fig9  = plot_2D(probit(u), xi, yi, title='Original Data')     # Plot true class assignments
# fig10 = plot_2D(t, xi[idx], yi[idx], title='Probit Data')     # Plot data
# fig9.show()
# fig10.show()

### ---- inferred field and thresholded version
# fig11 = plot_2D(Tinf, xi, yi, title='Inferred Data')
# fig11.show()
# fig12 = plot_2D(Tthres, xi, yi, title='Thresholded Data')
# fig12.show()

### ---- Heat map
# fig13 = plt.figure()
# ax13 = fig13.add_subplot()
# ax13.bar(x=lvals, height=grid_errors, alpha=0.6, edgecolor='black', lw=0.5, width=0.05)
# ax13.set_xlabel('length scale')
# ax13.set_ylabel('Mean absolute error')
# ax13.set_xticks([lvals[minidx], lvals[-1]])
# fig13.show()



### ---- pCN robustness - this is all done using the v generated on a 16x16 grid then subsampled by factor 4

# Dvals = np.array([16])
# Dvals = np.array([16, 26, 36, 46])
# betavals = np.logspace(np.log10(0.00001), np.log10(1.), 20)

# fig14 = plt.figure()
# ax14 = fig14.add_subplot()
# fig15 = plt.figure()
# ax15 = fig15.add_subplot()


# for Dval in tqdm(Dvals):
# 	grwacceptances = [] # containers
# 	pcnacceptances = []

# 	point = [(x, y) for y in np.arange(Dval) for x in np.arange(Dval)]                # Indexes for the inference grid
# 	coord = [(x, y) for y in np.linspace(0,1,Dval) for x in np.linspace(0,1,Dval)]    # Coordinates for the inference grid
# 	Nval = Dval*Dval # mesh size
# 	Kern = GaussianKernel(coord, l) + 1e-6*np.eye(Nval)# kernel on this mesh
# 	Gmat = get_G(Nval, idx) # Gmatrix for fixed M, increasing N
# 	Kernc = np.linalg.cholesky(Kern)
# 	Kernc_inverse = np.linalg.inv(Kernc)
# 	Kern_inverse = Kernc_inverse.T @ Kernc_inverse
# 	uprior = Kernc @ np.random.normal(size=Nval)

# 	for betaval in tqdm(betavals):

# 		_, agrw = grw(log_target, uprior, v, Kernc, Kern_inverse, Gmat, n, betaval)
# 		_, apcn = pcn(log_likelihood, uprior, v, Kernc, Kern_inverse, Gmat, n, betaval)

# 		grwacceptances.append(agrw)
# 		pcnacceptances.append(apcn)

# 	ax14.semilogx(betavals, grwacceptances, label=('D: '+str(Dval)))
# 	ax15.semilogx(betavals, pcnacceptances, label=('D: '+str(Dval)))

# ax14.set_xlabel('Step Size')
# ax14.set_ylabel('Acceptance Rate')
# ax15.set_xlabel('Step Size')
# ax15.set_ylabel('Acceptance Rate')
# fig14.suptitle('GRW-MH')
# fig15.suptitle('pCN')
# ax14.legend()
# ax15.legend()
# fig14.show()
# fig15.show()

### ---- Mean absolute error

# samps = 500
# tot1 = 0
# tot2 = 0
# for _ in tqdm(range(samps)):

# 	idx = subsample(N, subsample_factor)
# 	z = np.random.randn(N, )
# 	u = Kc @ z # data
# 	u0 = Kc @ np.random.normal(size=N) # prior
# 	G = get_G(N, idx)
# 	v = G @ u + np.random.randn(M)

# 	t = probit(v)

# 	Tpcn, accTpcn = pcn(log_probit_likelihood, u0, t, Kc, K_inverse, G, n, beta)
# 	Tinf = predict_t(Tpcn)
# 	Tthres = threshold(Tinf)

# 	prediction_error = np.mean(np.abs(Tthres-probit(u)))
# 	tot1 += prediction_error
# 	tot2 += accTpcn

# print("Mean error: "+str(tot1/samps))
# print("Acceptance: "+str(tot2/samps))
plt.show()