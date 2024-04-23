import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle





########################################################## Particle Class ########################################################## 
class Particle():
    def __init__(self, position, charge):#inverse mass?
        self.charge = charge
        self.position = position

        self.phi = 0


############################################################ Tree Class ############################################################ 
class bhm_tree():
    def __init__(self, centre_x, centre_y, half_width, half_height):

        self.centre_x = centre_x
        self.centre_y = centre_y


        self.half_width = half_width
        self.half_height = half_height

        self.particle = None
        self.children = []
        self.subdivided = False

        self.charge = 0
        self.charge_x = 0
        self.charge_y = 0

    def contains(self, particle):
        """Checks if the location of a particle is within the bounds of the section"""
        if particle.position[0] <= self.centre_x+self.half_width and particle.position[0] >= self.centre_x-self.half_width and particle.position[1] <= self.centre_y+self.half_height and particle.position[1] >= self.centre_y - self.half_height:
                return True
        return False

    def add_particle(self, particle):
        """Adds particle to quad tree - subdividing so that each contains maximum 1 particle"""

        if self.contains(particle): # checks if particle belongs in box - passed up to higher level

            if not self.subdivided and self.particle is None:
                #if the section does not already have a particle - add it and update com

                self.particle = particle

                self.charge_x = particle.position[0]
                self.charge_y = particle.position[1]
                self.charge = particle.charge
                return True

            elif not self.subdivided:
                #if not subdivided then must also contain a particle
                self.subdivide()

                # re-add the particle to a lower level
                to_add = self.particle
                self.particle = None
                self.charge_x = 0
                self.charge_y = 0
                self.charge = 0
                self.add_particle(to_add)
    
            
            for child in self.children:
                if child.add_particle(particle):
                    # for every particle contained by children update COM
                    self.charge_x = (self.charge_x*self.charge + particle.position[0]*particle.charge) / (self.charge + particle.charge)
                    self.charge_y = (self.charge_y*self.charge + particle.position[1]*particle.charge) / (self.charge + particle.charge)
                    self.charge+=particle.charge
                    return True


        return False

    def find_potential(self, particle, threshold = 0.5):
        distance = np.sqrt((particle.position[0] - self.charge_x)**2 + (particle.position[1] - self.charge_y)**2)
        if distance ==0:
            theta = threshold
        else:
            theta = self.half_width*2/distance 

        if theta < threshold:
            particle.phi-= self.charge*np.log(distance)

        elif self.subdivided:
            for child in self.children:
                child.find_potential(particle)
        elif self.particle != particle:
            particle.phi-= self.charge*np.log(distance)





    def subdivide(self):
        """Fills children list with 4 sub trees corresponding to each quadrant"""
        self.children.append(bhm_tree(self.centre_x+0.5*self.half_width, self.centre_y+0.5*self.half_height, self.half_width/2, self.half_height/2))
        self.children.append(bhm_tree(self.centre_x-0.5*self.half_width, self.centre_y+0.5*self.half_height, self.half_width/2, self.half_height/2))
        self.children.append(bhm_tree(self.centre_x+0.5*self.half_width, self.centre_y-0.5*self.half_height, self.half_width/2, self.half_height/2))
        self.children.append(bhm_tree(self.centre_x-0.5*self.half_width, self.centre_y-0.5*self.half_height, self.half_width/2, self.half_height/2))
        self.subdivided = True   


    def draw(self, ax):
        """draws the quad-tree, the particles contained and the COMs if desired"""
        ax.add_patch(Rectangle((self.centre_x-self.half_width, self.centre_y-self.half_height), self.half_width*2, 2*self.half_height, fill = False)) 
        # this shows the com of all sections - larger for larger charge in red
        if self.particle is not None:
            ax.scatter(*self.particle.position, marker=',', color='black', lw=0, s=4)
        else:
            ax.scatter(self.charge_x, self.charge_y, marker=',', color='red', lw=0, s=self.charge*15)
        # print(self.charge)
        if self.subdivided:
            for child in self.children:
                child.draw(ax)



########################################################## Running Algorithm ##########################################################

def bhm_potentials(positions, charges):
    """Creates tree, adds particles, then calculates and returns potential at each particle"""
    tree = bhm_tree(0.5, 0.5, 0.5, 0.5)

    particles = []
    for a in range(len(positions)):
        particle = Particle(positions[a], charges[a])
        particles.append(particle)
        tree.add_particle(particle)

    for particle in particles:
        tree.find_potential(particle)

    potentials = np.array([particle.phi for particle in particles])

    return potentials























