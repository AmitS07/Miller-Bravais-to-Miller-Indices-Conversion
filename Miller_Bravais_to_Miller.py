## Section-0: Importing Python Libraies
import numpy as np
## Section-1: Required Functions 
def angle_bw_two_vec(a,b):
    a_dot_b = np.dot(a,b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    cosine_ab = a_dot_b/(mag_a*mag_b)
    return cosine_ab   
    
## function to convert 4-index Miller-Bravias to 3-index Miller indices
def greatest_common_divisor(a, b):
    while b:  
        a, b = b, a % b  # Update 'a' to 'b' and 'b' to 'a % b'
    return a  

def miller_bravais_to_miller_direction(four_index_array):
    '''
    Input: 4-indicies should be enter either list or in array
    Output: List of 3-indicies
    Ref: Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    '''
    U,V,T,W = four_index_array[0], four_index_array[1], four_index_array[2], four_index_array[3]
    u = 2*U+V
    v = 2*V+U
    w = W
    uvw = [u,v,w]
    gcd = abs(uvw[0]) # assummed greatest_common_divisor
    for num in uvw[1:]:
        gcd = greatest_common_divisor(gcd, abs(num))
    if gcd > 0:
        vec = [val//gcd for val in uvw] # "//" integer divsion and "/" float divison
    else: 
        vec = uvw
    return vec

def miller_bravais_to_miller_plane(four_index_array):
    '''
    Input: 4-indicies should be enter either list or in array
    Output: List of 3-indicies
    Ref: Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    '''
    H,K,I,L = four_index_array[0], four_index_array[1], four_index_array[2], four_index_array[3]
    h = H
    k = K
    l = L
    return [h,k,l]

def cartesian_crystal_lattice(a,b,c,alpha,beta,gamma):
    '''
    Input: lattic parameters in order as a,b,c, alpha,beta,gamma
            Note: alpha, beta, gamma are angles in Degrees
    output: Certesian crystal lattice, a 3X3 matrix, a  transformation matrix
    Ref: Introduction to Texture analysis by Olaf Engler, Stefan Zaefferer and Valerie Randle
    '''
    # Convert angles in degree, since python defaults take in radian
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    ##
    l11, l12, l13 = a, b*np.cos(gamma), c*np.cos(beta) #  cosine of [100] to a, b, c axis of crystal, a|| [100]
    l21, l22, l23 = 0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/(np.sin(gamma)) # cosine of lattice a,b,c along crystal [010] axis
    l31, l32, l33 = 0, 0, c*np.sqrt(1+2*(np.cos(alpha)*np.cos(beta)*np.cos(gamma))-((np.cos(alpha)**2)+(np.cos(alpha)**2)+(np.cos(alpha)**2)))/(np.sin(gamma))
    crystal_lattice = np.array([[l11, l12, l13],
                               [l21, l22, l23],
                               [l31, l32, l33]])
    return crystal_lattice

#%% Section-2: Converting Miller-Bravais to Miller i.e. 4-indicies to 3-indices --------
## HCP slip-planes in Miller-Bravais
basal_plane = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1]])
prismatic_plane = np.array([[0,1,-1,0],[-1,0,1,0],[1,-1,0,0]])
pyramidal_plane = np.array([[1,0,-1,1],[1,0,-1,1],[0,1,-1,1],[0,1,-1,1],[-1,1,0,1],[-1,1,0,1],[-1,0,1,1],[-1,0,1,1],[0,-1,1,1],[0,-1,1,1],[1,-1,0,1],[1,-1,0,1]])
## HCP slip-direction in Miller-Bravais
a_direction = np.array([[2,-1,-1,0],[-1,2,-1,0],[-1,-1,2,0]])
c_a_direction = np.array([[-2,1,1,3],[-1,-1,2,3],[-1,-1,2,3],[1,-2,1,3],[1,-2,1,3],[2,-1,-1,3],[2,-1,-1,3],[1,1,-2,3],[1,1,-2,3],[-1,2,-1,3],[-1,2,-1,3],[-2,1,1,3]])

## Converting to 3-index notation
## HCP slip-planes in Miller
basal_miller = [miller_bravais_to_miller_plane(ii) for ii in basal_plane]
prism_miller = [miller_bravais_to_miller_plane(ii) for ii in prismatic_plane]
pyra_miller = [miller_bravais_to_miller_plane(ii) for ii in pyramidal_plane]
## HCP slip-direction in Miller
a_dir_miller = [miller_bravais_to_miller_direction(ii) for ii in a_direction]
c_a_dir_miller = [miller_bravais_to_miller_direction(ii) for ii in c_a_direction]

#%% Section-3: Making lattice orthonormal i.e. cartesian space ---------------------------------------------------------------

covera = 1.587 # for Titanium
## 3.1 For Direction 
# Defining Crsyatl-lattice matrix for HCP crystal, ref: Introduction to Texture analysis by Olaf Engler, Stefan Zaefferer and Valerie Randle
L_hcp = np.array([[1,-1/2,0],
		  [0,np.sqrt(3)/2,0],
		  [0,0,covera]])
## Note this L_HCP (Crystal-Lattice Matrix) can also be obtain by-
# L_HCP = cartesian_crystal_lattice(a,b,c,alpha,beta,gamma) # by defining hcP Lattice parameters

##for Direction:: Tranformimg HCP 3 index direction to Cartesian 3 indices
a_dir_cartFrame = [np.matmul(L_hcp, ii) for ii in a_dir_miller] # Miller Indices in Cartesian form, for <c+a> type slip direction
c_a_dir_cartFrame = [np.matmul(L_hcp, ii) for ii in c_a_dir_miller] # Miller indices in Cartesian form ,for <c+a> type slip direction
## Normalizing Directions
a_miller_norm = [ii/np.linalg.norm(ii) for ii in a_dir_cartFrame]
c_a_miller_norm = [ii/np.linalg.norm(ii) for ii in c_a_dir_cartFrame]

## 3.2: for plane-normals
# Defining L-Matrix (Crystal Lattice) for planes in Reciprocal lattice  i.e. Fourier Space or Reciprocal space (rcp.)
L_hcp_rcp = 2*np.pi*np.linalg.inv(L_hcp).T # Here 2*pi, to make sure priodicity in reciprocal space, Ref: Introduction to Solid-State Physics by Charles Kittel, 8th Edition
##for Plane:: Tranformimg HCP 3 indices planes to Cartesian 3 indices
basal_cartFrame = [np.matmul(L_HCP_REC,ii) for ii in basal_miller] # Miller Indices in Cartesian form, for Basal slip-planes
prism_cartFrame = [np.matmul(L_HCP_REC, ii) for ii in prism_miller] # Miller Indices in Cartesian form, for Prismatic slip-planes
pyra_cartFrame = [np.matmul(L_HCP_REC, ii) for ii in pyra_miller] # Miller Indices in Cartesian form, for Pyramidal slip-planes
## Normalizing Planes
basal_miller_norm = basal_cartFrame/np.linalg.norm(basal_cartFrame)
prism_miller_norm = [ii/np.linalg.norm(ii) for ii in prism_cartFrame]
pyra_miller_norm = [ii/np.linalg.norm(ii) for ii in pyra_cartFrame]

## To Check whether Direction lies in Planes or not i.e. hu +kv +lw = 0, i.e. angle between slip plane-normal to slip direction should be 90, i.e. cos90 = 0
## Testing for Pyramidal slip-system
angle_bw_plane_dir = [angle_bw_two_vec(ii,jj) for ii, jj in zip(pyra_miller_norm,c_a_miller_norm)]








