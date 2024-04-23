import numpy as np



def direct_method(positions, charges):
    number_particles = len(positions)
    potentials = np.zeros(number_particles)

    for a in range(number_particles):
        for b in range(number_particles):
            if a != b:
                r = np.sqrt((positions[a][0] - positions[b][0])**2 + (positions[a][1] - positions[b][1])**2)
                potentials[a] -= charges[b]*np.log(r)

    return potentials