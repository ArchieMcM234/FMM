from fmm import fmm_potentials
from barnes_hut import bhm_potentials, Particle
from direct_method import direct_method

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp




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



########################################################## Uniform Distribution ##########################################################

number_particles = 1024
positions = np.random.rand(number_particles,2)
charges = np.ones(number_particles)
num_terms = 20

bhm_phis = bhm_potentials(positions, charges)
fmm_phis = fmm_potentials(positions, charges, num_terms, 3)

direct_phis = direct_method(positions, charges)

bhm_error = np.mean((bhm_phis-direct_phis)/direct_phis)
fmm_error = np.mean((fmm_phis-direct_phis)/direct_phis)




print(f"{number_particles} uniformly distributed particles")
print(f"BHM relative error: {bhm_error:.10f}%")
print(f"FMM {num_terms} term expansion: {fmm_error:.10f}%")

########################################################## Plumer Model ##########################################################


def plumer_points(num_points, a):
    
    # Generate random radii using inverse transform method
    u = np.random.uniform(0, 1, num_points)
    radii = 0.1* np.sqrt(u**(-2/3) - 1)
    
    angles = np.random.uniform(0, 2*np.pi, num_points)
    # Convert polar coordinates to Cartesian coordinates
    x = 0.5 + radii * np.cos(angles)
    y = 0.5 + radii * np.sin(angles)
    positions = np.column_stack((x, y))
    return positions

attempt_number = 1024
positions = plumer_points(attempt_number, 0.1) # I found a=0.1 to work well



# will always be a probability of being outside the box - I chose to remove them
positions = positions[(positions[:, 0] >= 0) & (positions[:, 0] <= 1) &
                         (positions[:, 1] >= 0) & (positions[:, 1] <= 1)]


charges = np.ones(len(positions))


num_terms = 5

bhm_phis = bhm_potentials(positions, charges)
fmm_phis = fmm_potentials(positions, charges, num_terms)

direct_phis = direct_method(positions, charges)

bhm_error = np.mean((bhm_phis-direct_phis)/direct_phis)
fmm_error = np.mean((fmm_phis-direct_phis)/direct_phis)

print(f"{len(positions)} Plumer model distributed particles")
print(f"BHM relative error: {bhm_error:.10f}%")
print(f"FMM {num_terms} term expansion: {fmm_error:.10f}%")

fig, ax = plt.subplots()
ax.scatter(positions[:,0] , positions[:,1], marker=',', color='black', lw=0, s=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()

########################################################## FMM Convergence? ##########################################################




# fitted_slope, fitted_intercept, r_squared = linear_regression(np.log(sizes), avg_times/sizes)


# print(r_squared)



# # Plot the results
# plt.errorbar(np.log(sizes), avg_times/sizes, fmt='+', label='',  color='black', capsize=3,  markersize=4)
# plt.plot(np.log(sizes), linear_function(np.log(sizes), fitted_slope, fitted_intercept),color = 'black',linestyle = 'dashed', linewidth=1)
# plt.xlabel('log(N)', fontdict=font12)
# plt.ylabel('Time/N ($s^{-1}$)', fontdict=font12)
# plt.title('Complexity test for Barnes Hut algorithm', fontdict=font12)
# plt.show()







