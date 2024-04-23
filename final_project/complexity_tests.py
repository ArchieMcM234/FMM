from fmm import fmm_potentials
from barnes_hut import bhm_potentials

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import time


def linear_function(x, slope, intercept):
    return slope * x + intercept


def linear_regression(x, y):
    params, covariance = sp.optimize.curve_fit(linear_function, x, y)
    fitted_slope, fitted_intercept = params
    residuals = y- linear_function(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((x-np.mean(x))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return fitted_slope, fitted_intercept, r_squared

font12 = {'family': 'serif',
          'color':  'black',
          'weight': 'normal',
          'size': 12,
          }
font18 = {'family': 'serif',
          'color':  'black',
          'weight': 'normal',
          'size': 18,
          }





repeats = 1

sizes = np.array([500, 1000, 5000, 10000, 50000, 100000])
bhm_avg_times = np.array([])
fmm_avg_times = np.array([])
direct_avg_times = np.array([])

for number_particles in sizes:
    
    bhm_batch_times = np.array([])
    fmm_batch_times = np.array([])
    direct_batch_times = np.array([])
    for a in range(repeats):

        #generate fresh coordinates so average is representative
        positions = np.random.rand(number_particles,2)
        charges = np.ones(number_particles)



        start = time.time()
        bhm_potentials(positions, charges)
        bhm_batch_times = np.append(bhm_batch_times, time.time()-start)

        start = time.time()
        fmm_potentials(positions, charges, 5)
        fmm_batch_times = np.append(fmm_batch_times, time.time()-start)

        start = time.time()
        direct_method(positions, charges)
        direct_batch_times = np.append(bhm_batch_times, time.time()-start)


    #calculate and store averages and errors
    print('N:', number_particles, 'bhm:', np.mean(bhm_batch_times), 's fmm:', np.mean(fmm_batch_times), 's direct:', np.mean(direct_batch_times))
    bhm_avg_times = np.append(bhm_avg_times, np.mean(bhm_batch_times))
    fmm_avg_times = np.append(fmm_avg_times, np.mean(fmm_batch_times))
    direct_avg_times = np.append(direct_avg_times, np.mean(direct_batch_times))




bhm_slope, bhm_intercept, bhm_r_squared = linear_regression(np.log(sizes), bhm_avg_times/sizes)
fmm_slope, fmm_intercept, fmm_r_squared = linear_regression(sizes, fmm_avg_times)


print('bhm r_squared:', bhm_r_squared)
print('fmm r_squared:', fmm_r_squared)




# bhm graph
plt.scatter(np.log(sizes), bhm_avg_times/sizes, marker = '+', label='',  color='black',)
plt.plot(np.log(sizes), linear_function(np.log(sizes), bhm_slope, bhm_intercept),color = 'black',linestyle = 'dashed', linewidth=1)
plt.xlabel('log(N)', fontdict=font12)
plt.ylabel('Time/N ($s$)', fontdict=font12)
plt.title('Complexity test for Barnes Hut algorithm', fontdict=font12)
plt.show()

# fmm graph
plt.scatter(sizes, fmm_avg_times, marker = '+', label='',  color='black')
plt.plot(sizes, linear_function(sizes, fmm_slope, fmm_intercept),color = 'black',linestyle = 'dashed', linewidth=1)
plt.xlabel('N', fontdict=font12)
plt.ylabel('Time ($s$)', fontdict=font12)
plt.title('Complexity test for Fast Multipole Method', fontdict=font12)
plt.show()





