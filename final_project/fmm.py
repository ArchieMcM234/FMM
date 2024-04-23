import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import binom



######################################################## Expansion Methods ########################################################



def create_multipole(particle, centre_x, centre_y, num_terms):
    """Creates a multipole expansion about the centre of containing box"""

    coeffs = np.empty(num_terms + 1, dtype=complex) # coeffs can be complex

    coeffs[0] = particle.charge # this is the coefficient of the ln(z)
    z0 = complex(particle.position[0]-centre_x, particle.position[1] -centre_y)

    for k in range(1, num_terms+1):
        coeffs[k] = -particle.charge*z0**k/k

    return coeffs


def shift_multipole(coeffs, z0): 
    """Shifts multipoles from child nodes up the tree"""
    shifted = np.zeros_like(coeffs)
    shifted[0] = coeffs[0]

    for l in range(1, len(coeffs)):
        shifted[l] -= (coeffs[0]*z0**l)/l
        for k in range(1, l):         ##this might be a pinch point - could be to l but i think is right

            shifted[l] += coeffs[k]*z0**(l-k)*binom(l-1, k-1)

    return shifted


def multipole_to_local(coeffs, z0):
    """Creates a local (taylor) expansion from far field multipoles"""
    local = np.zeros_like(coeffs)
    local[0] = coeffs[0]*np.log(-z0)

    for k in range(1, len(coeffs)):
        local[0]+=(coeffs[k]/z0**k)*(-1)**k 
          

    for l in range(1, len(coeffs)):
        local[l] = -coeffs[0]/(l*z0**l)
        for k in range(1, len(coeffs)):
            local[l] += (1/z0**l)*(coeffs[k]/z0**k)*binom(l+k-1, k-1)*(-1)**k

    return local


def shift_local(coeffs, z0):
    """Moves the local (taylor) expansions down the tree"""
    shifted = np.zeros_like(coeffs)
    for l in range(len(coeffs)):
        for k in range(l, len(coeffs)):
            shifted[l]+=coeffs[k]*binom(k,l)*(-z0)**(k-l)
    return shifted



def direct_potential(particle1, particle2):
    """Calculates the potential from particle2 acting on particle1"""
    r = np.sqrt((particle1.position[0] - particle2.position[0])**2 + (particle1.position[1] - particle2.position[1])**2)
    return particle2.charge*np.log(r)


######################################################## Particle Class ########################################################
class Particle():
    def __init__(self, position, charge):#inverse mass?
        self.charge = charge
        self.position = position

        self.phi = 0





########################################################## Tree Class ##########################################################
class fmm_tree():
    def __init__(self, centre_x, centre_y, half_width, half_height, level, max_level, num_terms):

        self.centre_x = centre_x
        self.centre_y = centre_y

        self.half_width = half_width
        self.half_height = half_height

        self.particles = []
        self.children = []
        self.subdivided = False


        self.level = level
        self.max_level = max_level

        self.num_terms = num_terms

    def contains(self, particle):
        """Checks if a particle belongs within the bounds of the section"""
        if particle.position[0] <= self.centre_x+self.half_width and particle.position[0] >= self.centre_x-self.half_width and particle.position[1] <= self.centre_y+self.half_height and particle.position[1] >= self.centre_y - self.half_height:
            return True
        else:
            return False

    def subdivide(self):
        """Fills children list with 4 sub trees corresponding to each quadrant"""
        self.children.append(fmm_tree(self.centre_x+0.5*self.half_width, self.centre_y+0.5*self.half_height, self.half_width/2, self.half_height/2, self.level+1, self.max_level, self.num_terms))
        self.children.append(fmm_tree(self.centre_x-0.5*self.half_width, self.centre_y+0.5*self.half_height, self.half_width/2, self.half_height/2, self.level+1, self.max_level, self.num_terms))
        self.children.append(fmm_tree(self.centre_x+0.5*self.half_width, self.centre_y-0.5*self.half_height, self.half_width/2, self.half_height/2, self.level+1, self.max_level, self.num_terms))
        self.children.append(fmm_tree(self.centre_x-0.5*self.half_width, self.centre_y-0.5*self.half_height, self.half_width/2, self.half_height/2, self.level+1, self.max_level, self.num_terms)) 
        self.subdivided = True   

    def are_neighbours(self, other):
        """checks if two tree nodes are nearest neighbours"""
        if self.level== other.level:
            x_disp = abs(other.centre_x-self.centre_x)
            y_disp = abs(other.centre_y-self.centre_y)
            # tolerance added here to account for possible precision errors
            if x_disp <= self.half_width*2*1.3 and y_disp<=self.half_height*2*1.3: 
                return True
        return False 

    def create_tree(self):
        """Recursibvley subdivides the tree up to a maximum level"""
        if self.level < self.max_level:
            self.subdivide()
            for child in self.children:
                child.create_tree()


    def add_particle(self, particle):
        """Adds particle to quad tree to finest level of subtree - only subdivides those branches needed"""

        if self.contains(particle): # checks if particle belongs in box - passed up to higher level

            if self.level == self.max_level: 

                # add particle to leaf 
                self.particles.append(particle)
                return True

            for child in self.children:
                if child.add_particle(particle):
                    return True
        return False

    def upwards_pass(self):
        """recursively shifts the multipole expansion from children to parent using the multipole shift lemma"""
        self.multipole = np.zeros((self.num_terms + 1), dtype=complex)

        if not self.subdivided:
            for particle in self.particles:
                self.multipole += create_multipole(particle, self.centre_x, self.centre_y, self.num_terms)
        else:
            for child in self.children:
                child.upwards_pass()
                z0 = complex(child.centre_x, child.centre_y) - complex(self.centre_x, self.centre_y)
                self.multipole += shift_multipole(child.multipole, z0)


    def downwards_pass(self, parent = None, parent_neighbours = None):
        """Creats local expansions from far field sections, passed down local expansion throught tree and calculated final potentials"""

        # shift local expansion from parent to current node
        if parent is not None:
            z0 = complex(parent.centre_x, parent.centre_y) - complex(self.centre_x, self.centre_y) # check sign
            self.local = shift_local(parent.local, z0)
        else:
            self.local  = np.zeros(self.num_terms+1, dtype=complex)


        neighbours = []

        if parent_neighbours is None:
            #if this is the root node then need to include itself
            neighbours.append(self)
        else:
            # iterate through parents neighbours to find newly far field and nearest neighbours
            for parent_neighbour in parent_neighbours:
                for child in parent_neighbour.children:
                    if self.are_neighbours(child):
                        # if it is a direct neighbour then collect it
                        neighbours.append(child)
                    else:
                        # if in the far field then can include its influence as a local expansion
                        z0 = complex(child.centre_x, child.centre_y) - complex(self.centre_x, self.centre_y)
                        self.local += multipole_to_local(child.multipole, z0)

        # at this point we have a local expansion from the far field and a robust list of nearest neighbours            

        if self.subdivided:
            # if this node has children then carry on down 
            for child in self.children:
                child.downwards_pass(self, neighbours)

        else:
            z0 = complex(self.centre_x, self.centre_y)

            # iterate through this noded particles
            for particle in self.particles:
                z = complex(*particle.position)

                # evaluate the local expansion at the position of the particle
                particle.phi -= np.real(np.polyval(self.local[::-1], (z-z0)))
                # for particles in current and neighbouring nodes evaluate potential directly
                for neighbour in neighbours:
                    for other_particle in neighbour.particles:
                            if particle != other_particle:
                                particle.phi -= direct_potential(particle, other_particle)
            

    def draw(self, ax):
        """Draw tree to matplot lib axes - used for testing/visualisation"""

        ax.add_patch(Rectangle((self.centre_x-self.half_width, self.centre_y-self.half_height), self.half_width*2, 2*self.half_height, fill = False)) 

        if self.subdivided:
            for child in self.children:
                child.draw(ax)



def downwards_pass(node, parent=None, parent_neighbours=None):
    """Compute the inner expansions for all cells recursively and potential
    for all particles"""
    if parent is not None:
        z0 = complex(parent.centre_x, parent.centre_y) - complex(node.centre_x, node.centre_y) # check sign
        node.local = shift_local(parent.local, z0)
    else:
        node.local = np.zeros(node.num_terms+1, dtype=complex)
                

    neighbours = []
    if parent_neighbours is None:
        neighbours.append(node)
    else:
        for parent_neighbour in parent_neighbours:
            for child in parent_neighbour.children:
                if node.are_neighbours(child):
                    neighbours.append(child)
                else:
                    z0 = complex(child.centre_x, child.centre_y) - complex(node.centre_x, node.centre_y)
                    node.local += multipole_to_local(child.multipole, z0)

    if not node.subdivided:
        # Compute potential due to all far enough particles
        z0, coeffs = complex(node.centre_x, node.centre_y), node.local
        for particle in node.particles:
            z = complex(*particle.position)
            particle.phi -= np.real(np.polyval(coeffs[::-1], (z-z0)))
            # Compute potential directly from particles in interaction set
            for neighbour in neighbours:
                for other_particle in neighbour.particles:
                        if particle != other_particle:
                            particle.phi -= direct_potential(particle, other_particle)
    else:
        for child in node.children:
            downwards_pass(child, node, neighbours)





########################################################## Running Algorithm ##########################################################

def fmm_potentials(positions, charges, num_terms, max_level=None):
    """Creates tree, add particles, runs up and down passes and returns potentials at each particle"""

    #choose number of boxes so there is roughly one particle per box
    number_particles = len(positions)
    if max_level == None:
        max_level = np.round(np.log(number_particles) / np.log(4))

    tree = fmm_tree(0.5, 0.5, 0.5, 0.5, 0, max_level, num_terms)
    tree.create_tree()

    particles = []
    for a in range(number_particles):
        particle = Particle(positions[a], charges[a])
        particles.append(particle)
        tree.add_particle(particle)

    tree.upwards_pass()
    tree.downwards_pass()

    potentials = np.array([particle.phi for particle in particles])
    return potentials



