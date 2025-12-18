import numpy as np

sA=np.array([[]])  # the CV fluctuation of state A
sB=np.array([[]])  # the CV fluctuation of state B

N=2

muA=np.zeros(N)
muB=np.zeros(N)
for i in range(N): # the number of CVs
  muA[i] = np.mean(sA[i])
  muB[i] = np.mean(sB[i])
covA=np.cov(sA);covB=np.cov(sB)
cov_inv=np.linalg.inv(covA) + np.linalg.inv(covB)

between_class=np.outer((muA-muB),(muA-muB))
within_class_inv=cov_inv.copy()
tot_class=np.dot(within_class_inv,between_class)

wt, vt = np.linalg.eig(tot_class)
sidx = wt.argsort()[::-1] # sort according to the magnitude of the eigenvalues
wt = wt[sidx]
vt = vt[:,sidx]

wt = np.real(wt)
vt = np.real(vt)

#THIS IS A PRINTING TRICK!
vt = np.transpose(vt)

wt = wt.tolist()
vt = vt.tolist()

print(wt);print(vt)
