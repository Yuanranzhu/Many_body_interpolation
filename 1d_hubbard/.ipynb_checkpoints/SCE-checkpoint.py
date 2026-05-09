### The following strong coupling expansion is based on Yuan-Meztner's expansion. 
import itertools as it
import pickle
import numpy as np
from pydlr import dlr
import cvxpy as cp

def h0(Ns):
    """
    Construct the hopping matrix. If Ns%4 ==0 --> PBC, otherwise --> anti PBC 
    """
    h0 = np.zeros((Ns,Ns))
    for i in range(Ns):
        if i + 1 < Ns:  
            h0[i, i + 1] = -1.0 
        if i - 1 >= 0: 
            h0[i, i - 1] = -1.0
    if Ns%4 ==0:
        h0[0,-1] = -1.0 
        h0[-1,0] = -1.0
    else:
        h0[0,-1] = 1.0 
        h0[-1,0] = 1.0 
    return h0 


def dGij_sce_t(n, Ns, U, wn):
    """
    n-th Coeffs of SCE in terms of t  
    """
    iwn = 1j*wn
    t_mat = h0(Ns)
    dGij = np.zeros((wn.size, Ns, Ns),dtype='complex')
    if n > 3:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
    if n == 0: 
        for i in range(Ns):
            dGij[:, i, i] = iwn/((iwn)**2-U**2/4)
    elif n == 1:
        for i in range(Ns):
            for j in range(Ns):
                dGij[:, i, j] = -t_mat[i,j]*(iwn)**2/((iwn)**2-U**2/4)**2
    elif n == 2:
        t2 = t_mat@t_mat
        t2_diag = np.diag(np.diag(t2))
        term1 = (iwn)**3/((iwn)**2-U**2/4)**3
        term2 = (3/4)*iwn*U**2/((iwn)**2-U**2/4)**3
        for i in range(Ns):
            for j in range(Ns):
                dGij[:, i, j] = t2[i,j]*term1 + t2_diag[i,j]*term2
    return dGij
   
def Gij_sce_t(n, Ns, U, wn, t=1):
    """
    n-th order approximation of Gij_iwn using SCE in terms of t
    """
    G = np.zeros((wn.size, Ns, Ns),dtype='complex')
    t_mat = h0(Ns)

    Gij_sce_0 = dGij_sce_t(0, Ns, U, wn)
    Gij_sce_1 = dGij_sce_t(1, Ns, U, wn)
    Gij_sce_2 = dGij_sce_t(2, Ns, U, wn)
    
    if n > 3:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
    if n == 0: 
        Gij = Gij_sce_0
    elif n == 1:
        Gij = Gij_sce_0 + t*Gij_sce_1
    elif n == 2:
        Gij = Gij_sce_0 + t*Gij_sce_1 + t**2*Gij_sce_2
    return Gij 


# def Gij_sce(sce_order, Ns, U, wn):
#     """
#     n-th order approximation of Gij_iwn using SCE-Taylor series expansion
#     """
#     if sce_order > 3:
#         raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
#     Gij = (1.0/U) * Gij_sce_t(sce_order, Ns, 1.0/U, wn/U)
#     return Gij
            

# def Gij_sce_taylor(n, sce_order, Ns, U0, U, wn):
#     """
#     n-th order approximation of Gij_iwn using SCE-Taylor series expansion
#     """
#     Gij_iwn_0 = G0_FD(sce_order, Ns, U0, wn)
#     Gij_iwn_1 = G1_FD(sce_order, Ns, U0, wn)
#     Gij_iwn_2 = G2_FD(sce_order, Ns, U0, wn)
#     if n > 3:
#         raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")  
#     if n==0: 
#         Gij = Gij_iwn_0
#     if n==1:
#         Gij = Gij_iwn_0 + (U-U0)*Gij_iwn_1
#     if n==2:
#         Gij = Gij_iwn_0 + (U-U0)*Gij_iwn_1+ 0.5*(U-U0)**2*Gij_iwn_2
#     return Gij
        
