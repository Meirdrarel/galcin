import numpy as np
import pymultinest
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as AxesD
from astropy.io import ascii

output = pymultinest.Analyzer(n_params=7, outputfiles_basename='../test/1500317/multinest/output_')
data = output.get_data()
shapeD = np.shape(data)
# stats = output.get_mode_stats()
# bestfit = output.get_best_fit()

lnZ = data[:, 0]
x = data[:, 2]
y = data[:, 3]
pa = data[:, 4]
incl = data[:, 5]
vs = data[:, 6]
vm = data[:, 7]
rd = data[:, 8]


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(x, y, lnZ)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('ln(Z)')
plt.draw()


fig2 = plt.figure(2)
plt.plot(pa, lnZ)
plt.draw()

plt.show()
