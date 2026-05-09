### The following strong coupling expansion for 2D Hubbard model is based on Yuan-Meztner's expansion. 
import itertools as it
import pickle
import numpy as np
from pydlr import dlr
import cvxpy as cp

def generate_hopping_matrix(L, HoppingT=1.0):
    """
    Generate the hopping matrix for a sqrt(N)*sqrt(N) 2D lattice Hubbard model
    with periodic boundary conditions.

    Parameters:
        L*L(int): size of the 2D lattice.

    Returns:
        np.ndarray: Hopping matrix of size (N, N).
    """
    # Number of sites
    num_sites = L
    N = L**2 # Total lattice size
    # print("The number of sites are", num_sites)
    # Initialize the hopping matrix with zeros
    hopping_matrix = np.zeros((N, N))

    def site_index(x, y):
        """
        Map 2D lattice coordinates (x, y) to 1D index.

        Parameters:
            x (int): x-coordinate on the lattice.
            y (int): y-coordinate on the lattice.

        Returns:
            int: 1D index corresponding to the site.
        """
        return (x % num_sites) + (y % num_sites) * num_sites

    # Loop over each site in the 2D lattice
    for x in range(num_sites):
        for y in range(num_sites):
            # Current site index
            current_site = site_index(x, y)

            # Nearest neighbors with periodic boundary conditions
            neighbors = [
                site_index(x + 1, y),  # Right neighbor
                site_index(x - 1, y),  # Left neighbor
                site_index(x, y + 1),  # Top neighbor
                site_index(x, y - 1)   # Bottom neighbor
            ]

            # Set hopping values (-1 for nearest neighbors)
            for neighbor in neighbors:
                hopping_matrix[current_site, neighbor] = HoppingT

    return hopping_matrix


def dGij_sce_t(n, Ns, U, wn):
    """
    n-th Coeffs of SCE in terms of t  
    """
    iwn = 1j*wn
    t_mat = generate_hopping_matrix(Ns)
    dGij = np.zeros((wn.size, Ns**2, Ns**2),dtype='complex')
    if n > 3:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
    if n == 0: 
        for i in range(Ns**2):
            dGij[:, i, i] = iwn/((iwn)**2-U**2/4)
    elif n == 1:
        for i in range(Ns**2):
            for j in range(Ns**2):
                dGij[:, i, j] = -t_mat[i,j]*(iwn)**2/((iwn)**2-U**2/4)**2
    elif n == 2:
        t2 = t_mat@t_mat
        t2_diag = np.diag(np.diag(t2))
        term1 = (iwn)**3/((iwn)**2-U**2/4)**3
        term2 = (3/4)*iwn*U**2/((iwn)**2-U**2/4)**3
        for i in range(Ns**2):
            for j in range(Ns**2):
                dGij[:, i, j] = t2[i,j]*term1 + t2_diag[i,j]*term2
    return dGij
   
def G_sce(n, Nkx, Nky, U, t, wn, beta):
    """
    n-th order approximation of Gij_iwn using SCE in terms of t, U=1.0 
    """
    G = np.zeros((wn.size, Nkx*Nky,Nkx*Nky),dtype='complex')
    t_mat = generate_hopping_matrix(Nkx)

    Gij_sce_0 = dGij_sce_t(0, Nkx, U, wn)
    Gij_sce_1 = dGij_sce_t(1, Nkx, U, wn)
    Gij_sce_2 = dGij_sce_t(2, Nkx, U, wn)
    
    if n > 3:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
    if n == 0: 
        Gij = Gij_sce_0
    elif n == 1:
        Gij = Gij_sce_0 + t*Gij_sce_1
    elif n == 2:
        Gij = Gij_sce_0 + t*Gij_sce_1 + t**2*Gij_sce_2
    return Gij 

        
