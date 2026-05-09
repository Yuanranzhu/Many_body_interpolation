### This gathers the utilities function for Pade[N/N] approximation for Green's function of Hubbard dimer
import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pade
from math import comb
from math import factorial
from scipy.linalg import svd

########### WCE expansion Coeffs
def G0_ij_wce(wn, t):
    iwn = 1j*wn
    b1 = iwn + t
    b2 = iwn - t
    b3 = iwn - t
    b4 = iwn + t
    G0 = np.zeros((wn.size, 2,2), dtype='complex')
    G0[:,0,0] = 0.5*(1/(b1-2*t)+1/(b2+2*t))
    G0[:,0,1] = 0.5*(1/(2*t-b1)+1/(b2+2*t))
    G0[:,1,0] = 0.5*(1/(2*t-b1)+1/(b2+2*t))
    G0[:,1,1] = 0.5*(1/(b1-2*t)+1/(b2+2*t))
    return G0

def G2_ij_wce(wn, t):
    iwn = 1j*wn
    b1 = iwn + t
    b2 = iwn - t
    b3 = iwn - t
    b4 = iwn + t
    G2 = np.zeros((wn.size, 2,2), dtype='complex')
    G2[:,0,0] =  0.5*((-b1+6*t)/(64*(b1-2*t)**2*t**2)-(b2+6*t)/(64*(b2+2*t)**2*t**2)+1/(64*(b3-2*t)*t**2)+ 1/(64*(b4+2*t)*t**2))
    G2[:,0,1] =  0.5*((b1-6*t)/(64*(b1-2*t)**2*t**2)-(b2+6*t)/(64*(b2+2*t)**2*t**2)+1/(64*(b3-2*t)*t**2)- 1/(64*(b4+2*t)*t**2))
    G2[:,1,0] =  0.5*((b1-6*t)/(64*(b1-2*t)**2*t**2)-(b2+6*t)/(64*(b2+2*t)**2*t**2)+1/(64*(b3-2*t)*t**2)- 1/(64*(b4+2*t)*t**2))
    G2[:,1,1] =  0.5*((-b1+6*t)/(64*(b1-2*t)**2*t**2)-(b2+6*t)/(64*(b2+2*t)**2*t**2)+1/(64*(b3-2*t)*t**2)+ 1/(64*(b4+2*t)*t**2))
    return G2

def G4_ij_wce(wn, t):
    iwn = 1j*wn
    G4 = np.zeros((wn.size, 2,2), dtype='complex')
    b1 = iwn + t
    b2 = iwn - t
    b3 = iwn - t
    b4 = iwn + t
    G4[:,0,0] = 0.5*((3*b1**2-20*b1*t+44*t**2)/(4096*(b1-2*t)**3*t**4) + (3*b2**2+20*b2*t+44*t**2)/(4096*(b2+2*t)**3*t**4)  + (10*t-3*b3)/(4096*(b3-2*t)**2*t**4) - (10*t+3*b4)/(4096*(b4+2*t)**2*t**4) )
    G4[:,0,1] = 0.5*(-(3*b1**2-20*b1*t+44*t**2)/(4096*(b1-2*t)**3*t**4) + (3*b2**2+20*b2*t+44*t**2)/(4096*(b2+2*t)**3*t**4)  + (10*t-3*b3)/(4096*(b3-2*t)**2*t**4) + (10*t+3*b4)/(4096*(b4+2*t)**2*t**4) )
    G4[:,1,0] = G4[:,0,1]
    G4[:,1,1] = G4[:,0,0]
    return G4


def G6_ij_wce(wn, t):
    iwn = 1j*wn
    b1 = iwn + t
    b2 = iwn - t
    b3 = iwn - t
    b4 = iwn + t
    G6 = np.zeros((wn.size, 2,2), dtype='complex')
    G6[:,0,0] = 0.5*(-(5*b1**3-42*b1**2*t+132*b1*t**2-168*t**3)/(131071*(b1-2*t)**4*t**6) - (5*b2**3+42*b2**2*t+132*b2*t**2+168*t**3)/(131072*(b2+2*t)**4*t**6)  + (44*t**2-28*b3*t+5*b3**2)/(131072*(b3-2*t)**3*t**6)+ (44*t**2+28*b4*t+5*b4**2)/(131072*(b4+2*t)**3*t**6) )
    G6[:,1,1] = G6[:,0,0]
    G6[:,0,1] = 0.5*((5*b1**3-42*b1**2*t+132*b1*t**2-168*t**3)/(131071*(b1-2*t)**4*t**6) - (5*b2**3+42*b2**2*t+132*b2*t**2+168*t**3)/(131072*(b2+2*t)**4*t**6)  + (44*t**2-28*b3*t+5*b3**2)/(131072*(b3-2*t)**3*t**6)- (44*t**2+28*b4*t+5*b4**2)/(131072*(b4+2*t)**3*t**6) )  
    G6[:,1,0] = G6[:,0,1] 
    return G6

# (35 iw^4 - 220 iw^3 t + 610 iw^2 t^2 - 956 iw t^3 + 787 t^4) /33554432 (iw - t)^5 t^8
# (35 iw^4 + 220 iw^3 t + 610 iw^2 t^2 + 956 iw t^3 + 787 t^4) /33554432 (iw + t)^5 t^8
# (35 iw^3 - 375 iw^2 t + 1385 iw t^2 - 1789 t^3) /33554432 (iw - 3 t)^4 t^8
# (35 iw^3 + 375 iw^2 t + 1385 iw t^2 + 1789 t^3) /33554432 (iw + 3 t)^4 t^8
def G8_ij_wce(wn, t):
    iwn = 1j*wn
    G8 = np.zeros((wn.size, 2,2), dtype='complex')
    term1 = (35*iwn**4-220*iwn**3*t+610*iwn**2*t**2-956*iwn*t**3+787*t**4)/(33554432*(iwn-t)**5*t**8)
    term2 = (35*iwn**4+220*iwn**3*t+610*iwn**2*t**2+956*iwn*t**3+787*t**4)/(33554432*(iwn+t)**5*t**8)
    term3 = -(35*iwn**3-375*iwn**2*t+1385*iwn*t**2-1789*t**3)/(33554432*(iwn-3*t)**4*t**8)
    term4 = -(35*iwn**3+375*iwn**2*t+1385*iwn*t**2+1789*t**3)/(33554432*(iwn+3*t)**4*t**8)
    G8[:,0,0] = term1 + term2 + term3 + term4
    G8[:,1,1] = G8[:,0,0]
    G8[:,0,1] = -term1 + term2 + term3 - term4
    G8[:,1,0] = G8[:,0,1] 
    return G8

# -(63 iw^5 - 455 iw^4 t + 1470 iw^3 t^2 - 2790 iw^2 t^3 + 3315 iw t^4 - 2115 t^5)/1073741824 (iw - t)^6 t^10
# -(63 iw^5 + 455 iw^4 t + 1470 iw^3 t^2 + 2790 iw^2 t^3 + 3315 iw t^4 + 2115 t^5)/1073741824 (iw - t)^6 t^10
# (63 iw^4 - 868 iw^3 t + 4578 iw^2 t^2 - 11028 iw t^3 + 10343 t^4)/1073741824 (iw - 3t)^5 t^10
# (63 iw^4 + 868 iw^3 t + 4578 iw^2 t^2 + 11028 iw t^3 + 10343 t^4)/1073741824 (iw - 3t)^5 t^10
def G10_ij_wce(wn, t):
    iwn = 1j*wn
    G10 = np.zeros((wn.size, 2,2), dtype='complex')
    term1 = -(63*iwn**5-455*iwn**4*t+1470*iwn**3*t**2-2790*iwn**2*t**3+3315*iwn*t**4-2115*t**5)/(1073741824*(iwn-t)**6*t**10)
    term2 = -(63*iwn**5+455*iwn**4*t+1470*iwn**3*t**2+2790*iwn**2*t**3+3315*iwn*t**4+2115*t**5)/(1073741824*(iwn+t)**6*t**10)
    term3 = (63*iwn**4-868*iwn**3*t+4578*iwn**2*t**2-11028*iwn*t**3+10343*t**4)/(1073741824*(iwn-3*t)**5*t**10)
    term4 = (63*iwn**4+868*iwn**3*t+4578*iwn**2*t**2+11028*iwn*t**3+10343*t**4)/(1073741824*(iwn+3*t)**5*t**10)
    G10[:,0,0] = term1 + term2 + term3 + term4
    G10[:,1,1] = G10[:,0,0]
    G10[:,0,1] = -term1 + term2 + term3 - term4
    G10[:,1,0] = G10[:,0,1] 
    return G10

# (231 iw^6 - 1890 iw^5 t + 6993 iw^4 t^2 - 15484 iw^3 t^3 + 22617 iw^2 t^4 - 21762 iw t^5 + 11343 t^6)/68719476736 (iw - t)^7 t^12
# (231 iw^6 + 1890 iw^5 t + 6993 iw^4 t^2 + 15484 iw^3 t^3 + 22617 iw^2 t^4 + 21762 iw t^5 + 11343 t^6)/68719476736 (iw + t)^7 t^12
# (-231 iw^5 + 3885 iw^4 t - 26502 iw^3 t^2 + 91994 iw^2 t^3 - 163331 iw t^4 + 119561 t^5)/ 68719476736 (iw - 3 t)^6 t^12
# (231 iw^5 + 3885 iw^4 t + 26502 iw^3 t^2 + 91994 iw^2 t^3 + 163331 iw t^4 + 119561 t^5)/ 68719476736 (iw + 3 t)^6 t^12
def G12_ij_wce(wn, t):
    iwn = 1j*wn
    G12 = np.zeros((wn.size, 2,2), dtype='complex')
    term1 = (231*iwn**6-1890*iwn**5*t+6993*iwn**4*t**2-15484*iwn**3*t**3+22617*iwn**2*t**4-21762*iwn*t**5+ 11343*t**6)/(68719476736*(iwn-t)**7*t**12)
    term2 = (231*iwn**6+1890*iwn**5*t+6993*iwn**4*t**2+15484*iwn**3*t**3+22617*iwn**2*t**4+21762*iwn*t**5+ 11343*t**6)/(68719476736*(iwn+t)**7*t**12)
    term3 = (-231*iwn**5+3885*iwn**4*t-26502*iwn**3*t**2+91994*iwn**2*t**3-163331*iwn*t**4+ 119561*t**5)/(68719476736*(iwn-3*t)**6*t**12)
    term4 = (231*iwn**5+3885*iwn**4*t+26502*iwn**3*t**2+91994*iwn**2*t**3+163331*iwn*t**4+ 119561*t**5)/(68719476736*(iwn+3*t)**6*t**12)
    G12[:,0,0] = term1 + term2 + term3 + term4
    G12[:,1,1] = G12[:,0,0]
    G12[:,0,1] = -term1 + term2 + term3 - term4
    G12[:,1,0] = G12[:,0,1] 
    return G12

    
############# WCE expansion in terms of U
def Gij_wce_series(n, t, U, wn):
    """
    Exact series expansion for Green's function
    
    Parameters:
    - U (float): Interaction strength.
    - wn (1D array): Matsubara frequencies.
    
    Returns:
    - Gn (2D array): Reconstructed Green's function [Nw* Ns* Ns].
    """
    if n > 12:
        raise ValueError("Expansion order n is too large! Only n <= 12 is supported.")
    
    # Calculate the approximated Green's function
    if n == 0:
        Gn = G0_ij_wce(wn, t)
    elif n == 1:
        Gn = G0_ij_wce(wn, t)
    elif n == 2:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t)
    elif n == 3:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t)
    elif n == 4:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t)
    elif n == 5:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t)
    elif n == 6:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)
    elif n == 7:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)
    elif n == 8:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)+ U**8*G8_ij_wce(wn, t)
    elif n == 9:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)+ U**8*G8_ij_wce(wn, t)
    elif n == 10:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)+ U**8*G8_ij_wce(wn, t)+ U**10*G10_ij_wce(wn, t)
    elif n == 11:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)+ U**8*G8_ij_wce(wn, t)+ U**10*G10_ij_wce(wn, t)
    elif n == 12:
        Gn = G0_ij_wce(wn, t) + U**2*G2_ij_wce(wn, t) + U**4*G4_ij_wce(wn, t) + U**6*G6_ij_wce(wn, t)+ U**8*G8_ij_wce(wn, t)+ U**10*G10_ij_wce(wn, t)+ U**12*G12_ij_wce(wn, t)
    return Gn

########################### SCE expansion of t
def G0_ij_sce(wn, U, beta):
    G0_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G0_ij[:,0,0] = iwn/((iwn)**2-U**2/4)
    G0_ij[:,1,1] = iwn/((iwn)**2-U**2/4)
    return G0_ij

def G1_ij_sce(wn, U, beta):
    """
    08/28 updated, try to add 3/4 additional term to ensure the right asymptotics 
    """
    G1_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G1_ij[:,0,1] = -(iwn)**2/((iwn)**2-U**2/4)**2 
    G1_ij[:,1,0] = -(iwn)**2/((iwn)**2-U**2/4)**2 
    return G1_ij

def G2_ij_sce(wn, U, beta):
    G2_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G2_ij[:,0,0] = (iwn)**3/((iwn)**2-U**2/4)**3 + 3/4*U**2*iwn/((iwn)**2-U**2/4)**3
    G2_ij[:,1,1] = (iwn)**3/((iwn)**2-U**2/4)**3 + 3/4*U**2*iwn/((iwn)**2-U**2/4)**3
    return G2_ij


def G3_ij_sce(wn, U, beta):
    G3_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn

    A = (-4 * iwn**2 + U**2)

    num = (
        -96 * beta * iwn**2 * U**3
        + 12 * U**4 * (-4 + beta * U)
        + 64 * iwn**4 * (-4 + 3 * beta * U)
    )

    den = A**4

    # G3_ij[:,0,1] = -16*(16*iwn**4+3*U**4)/(-4*iwn**2+U**2)**4 +12*beta*U/(-4*iwn**2+U**2)**4 
    G3_ij[:,0,1] = num / den
    G3_ij[:,1,0] = G3_ij[:,0,1]
    return G3_ij

def G4_ij_sce(wn, U, beta):
    G4_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G4_ij[:,0,0] = 64*(16*iwn**5+72*iwn**3*U**2-3*iwn*U**4)/(4*iwn**2-U**2)**5
    G4_ij[:,1,1] = G4_ij[:,0,0]
    return G4_ij


def G5_ij_sce(wn, U, beta):
    G5_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    # G5_ij[:,0,1] = -128*(32*iwn**6-192*iwn**4*U**2+90*iwn**2*U**4-3*U**6)/(-4*iwn**2+U**2)**6
    # G5_ij[:,0,1] = G5_ij[:,0,1]+(12*beta*(-8*iwn**2*U**2*(-16+beta*U)+U**4*(-12+beta*U)+ 16*iwn**4*(-4+beta*U))) / (U*(-4*iwn**2+U**2)**4)

    A = (-4 * iwn**2 + U**2)

    num = 4 * (
        3 * beta**2 * U * A**4
        - 12 * beta * A**2 * (16 * iwn**4 - 32 * iwn**2 * U**2 + 3 * U**4)
        - 32 * (32 * iwn**6 * U - 192 * iwn**4 * U**3 + 90 * iwn**2 * U**5 - 3 * U**7)
    )

    den = U * (A**6)
    G5_ij[:,0,1] = num / den
    G5_ij[:,1,0] = G5_ij[:,0,1]
    return G5_ij

def G6_ij_sce(wn, U, beta):
    G6_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    B = (4 * iwn**2 - U**2)

    inner = (
        144 * iwn**4 * U**2 * (13 - 2 * beta * U)
        + 3 * U**6 * (13 - 2 * beta * U)
        + 12 * iwn**2 * U**4 * (-31 + 6 * beta * U)
        + 64 * iwn**6 * (1 + 6 * beta * U)
    )

    num = 256 * iwn * inner
    den = B**7
    
    G6_ij[:,0,0] = num/ den
    # G6_ij[:,0,0] = 256*(64*iwn**7+1872*iwn**5*U**2-372*iwn**3*U**4+39*iwn*U**6)/(4*iwn**2-U**2)**7
    G6_ij[:,1,1] = G6_ij[:,0,0]
    return G6_ij

def G7_ij_sce(wn, U, beta):
    G7_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    A = (-4 * iwn**2 + U**2)

    num = (
        4 * (
            beta**3 * U**2 * A**6
            + 48 * beta**2 * U * A**4 * (8 * iwn**4 - 10 * iwn**2 * U**2 + U**4)
            - 48 * beta * A**2 * (
                512 * iwn**8
                - 896 * iwn**6 * U**2
                + 1040 * iwn**4 * U**4
                - 192 * iwn**2 * U**6
                + 11 * U**8
            )
            + 64 * U**3 * (
                256 * iwn**8
                - 18432 * iwn**6 * U**2
                + 7968 * iwn**4 * U**4
                - 672 * iwn**2 * U**6
                + 21 * U**8
            )
        )
    )

    den = (U**3) * (A**8)
    G7_ij[:,0,1] = -(num / den)
    # G7_ij[:,0,1] = -256*(256*iwn**8-18432*iwn**6*U**2+7968*iwn**4*U**4-672*iwn**2*U**6+21*U**8)/(-4*iwn**2+U**2)**8
    G7_ij[:,1,0] = G7_ij[:,0,1]
    return G7_ij

def G8_ij_sce(wn, U, beta):
    G8_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    A = (-4 * iwn**2 + U**2)   # (-4 iw^2 + u^2)
    B = (4 * iwn**2 - U**2)    # (4 iw^2 - u^2) = -A

    inner = (
        512 * iwn**8 * U
        + 121344 * iwn**6 * U**3
        - 46656 * iwn**4 * U**5
        + 7200 * iwn**2 * U**7
        - 270 * U**9
        - 3 * beta**2 * U * (A**5)
        - 12 * beta * (B**3) * (16 * iwn**4 - 56 * iwn**2 * U**2 + 5 * U**4)
    )

    num = -512 * iwn * inner
    den = U * (A**9)
    G8_ij[:,0,0] = num/den
    # G8_ij[:,0,0] = 1024*(256*iwn**9+60672*iwn**7*U**2-23328*iwn**5*U**4+3600*iwn**3*U**6-135*iwn*U**8)/(4*iwn**2-U**2)**9
    G8_ij[:,1,1] = G8_ij[:,0,0]
    return G8_ij

def G9_ij_sce(wn, U, beta):
    G9_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G9_ij[:,0,1] = -4096*(256*iwn**10-175104*iwn**8*U**2+88416*iwn**6*U**4-13680*iwn**4*U**6+945*iwn**2*U**8-15*U**10)/(-4*iwn**2+U**2)**10
    G9_ij[:,1,0] = G9_ij[:,0,1]
    return G9_ij

def G10_ij_sce(wn, U, beta):
    G10_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G10_ij[:,0,0] = 4096*(1024*iwn**11+2135808*iwn**9*U**2-1146240*iwn**7*U**4+234720*iwn**5*U**6-18540*iwn**3*U**8+555*iwn*U**10)/(4*iwn**2-U**2)**11
    G10_ij[:,1,1] = G10_ij[:,0,0]
    return G10_ij  

def G11_ij_sce(wn, U, beta):
    G11_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G11_ij[:,0,1] = -4096*(4096*iwn**12-25460736*iwn**10*U**2+15641856*iwn**8*U**4-3506688*iwn**6*U**6+373680*iwn**4*U**8-15840*iwn**2*U**10+183*U**12)/(-4*iwn**2+U**2)**12
    G11_ij[:,1,0] = G11_ij[:,0,1]
    return G11_ij
    
def G12_ij_sce(wn, U, beta):
    G12_ij = np.zeros((wn.size, 2,2), dtype='complex')
    iwn = 1j*wn
    G12_ij[:,0,0] = 16385*(4096*iwn**13+76584960*iwn**11*U**2-51277056*iwn**9*U**4+13420800*iwn**7*U**6-1635984*iwn**5*U**8+96600*iwn**3*U**10-1995*iwn*U**12)/(4*iwn**2-U**2)**12
    G12_ij[:,1,1] = G12_ij[:,0,0]
    return G12_ij  
    
############# WCE expansion in terms of t

def Gij_sce_series(n, t, U, wn, beta):
    """
    Exact series expansion for matrix Green's function G_ij(iwn)
    
    Parameters:
    - U (float): Interaction strength.
    - wn (1D array): Matsubara frequencies.
    
    Returns:
    - Gn (2D array): Reconstructed Green's function [Nw* Ns* Ns].
    """
    if n > 12:
        raise ValueError("Expansion order n is too large! Only n <= 12 result is supported.")
    
    # Calculate the approximated Green's function
    if n == 0:
        Gn_ij = G0_ij_sce(wn, U, beta)
    elif n == 1:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta)
    elif n == 2:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)
    elif n == 3:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)
    elif n == 4:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)
    elif n == 5:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)
    elif n == 6:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)
    elif n == 7:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)
    elif n == 8:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)+ t**8*G8_ij_sce(wn, U, beta)
    elif n == 9:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)+ t**8*G8_ij_sce(wn, U, beta)+t**9*G9_ij_sce(wn, U, beta)
    elif n == 10:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)+ t**8*G8_ij_sce(wn, U, beta)+t**9*G9_ij_sce(wn, U, beta)+t**10*G10_ij_sce(wn, U, beta)
    elif n == 11:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)+ t**8*G8_ij_sce(wn, U, beta)+t**9*G9_ij_sce(wn, U, beta)+t**10*G10_ij_sce(wn, U, beta)+t**11*G11_ij_sce(wn, U, beta)
    elif n == 12:
        Gn_ij = G0_ij_sce(wn, U, beta)+ t*G1_ij_sce(wn, U, beta) + t**2*G2_ij_sce(wn, U, beta)+ t**3*G3_ij_sce(wn, U, beta)+ t**4*G4_ij_sce(wn, U, beta)+ t**5*G5_ij_sce(wn, U, beta)+ t**6*G6_ij_sce(wn, U, beta)+ t**7*G7_ij_sce(wn, U, beta)+ t**8*G8_ij_sce(wn, U, beta)+t**9*G9_ij_sce(wn, U, beta)+t**10*G10_ij_sce(wn, U, beta)+t**11*G11_ij_sce(wn, U, beta)+t**12*G12_ij_sce(wn, U, beta)
    return Gn_ij




# def G0_FD_wce(n, t0, U0, wn, k):
#     G0_ij = Gij_wce_series(n, t0, U0, wn)
#     return G0_ij
    

# def G1_FD_wce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order central finite-difference for the first derivative f'(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float or complex
#     t0 : float
#     h : float, step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order of the scheme (default 4)

#     Returns
#     -------
#     complex
#         Approximation to f'(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     if order == 2:
#         # O(h^2) 3-point central
#         fp = f(t0 - k*h, U0+h)
#         fm = f(t0 + k*h, U0-h)
#         return (fp - fm) / (2*h)

#     elif order == 4:
#         # O(h^4) 5-point central
#         fp1 = f(t0 - k*h , U0+h);   fm1 = f(t0 + k*h, U0-h)
#         fp2 = f(t0 - 2*k*h, U0+ 2*h); fm2 = f(t0 + 2*k*h, U0-2*h)
#         return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12*h)

#     elif order == 6:
#         # O(h^6) 7-point central
#         fp1 = f(t0 -k*h, U0+h);   fm1 = f(t0 + k*h, U0-h)
#         fp2 = f(t0 - 2*k*h, U0+2*h); fm2 = f(t0 + 2*k*h, U0-2*h)
#         fp3 = f(t0 - 3*k*h, U0+3*h); fm3 = f(t0 + 3*k*h, U0-3*h)
#         return (fm3 - 9*fm2 + 45*fm1 - 45*fp1 + 9*fp2 - fp3) / (60*h)

#     else:
#         raise ValueError("order must be one of {2, 4, 6}")

# def G2_FD_wce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order central finite-difference for the second derivative f''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f''(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     # # Ensure we don't evaluate at U=0
#     # for k in range(1, (order//2)*2):  # k=1,2 for order>=4; k=1..3 for order=6
#     #     if np.isclose(t0 + k*h, 0.0) or np.isclose(t0 - k*h, 0.0):
#     #         raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 3-point central, O(h^2)
#         fp1 = f(t0 - k*h, U0 + h); f0 = f(t0, U0); fm1 = f(t0 + k*h, U0 - h)
#         return (fp1 - 2*f0 + fm1) / (h*h)

#     elif order == 4:
#         # 5-point central, O(h^4)
#         fp2 = f(t0 - 2*k*h, U0 + 2*h); fp1 = f(t0 - k*h, U0 + h)
#         f0  = f(t0, U0)
#         fm1 = f(t0 + k*h, U0 - h);   fm2 = f(t0 + 2*k*h, U0 - 2*h)
#         return (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*h*h)
        
#     elif order == 6:
#         # 7-point central, O(h^6)
#         fp3 = f(t0 - 3*k*h, U0 + 3*h); fp2 = f(t0 - 2*k*h, U0 + 2*h); fp1 = f(t0 - k*h, U0 + h)
#         f0  = f(t0, U0 )
#         fm1 = f(t0 + k*h, U0 - h);   fm2 = f(t0 + 2*k*h, U0 - 2*h); fm3 = f(t0 + 3*k*h, U0 - 3*h)
#         return (-fm3/90 + 3*fm2/20 - 3*fm1/2 + 49*f0/18
#                 - 3*fp1/2 + 3*fp2/20 - fp3/90) / (h*h)

#     else:
#         raise ValueError("order must be one of {2, 4, 6}")


# def G3_FD_wce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order centered finite-difference for f'''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f'''(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     # Avoid evaluating at U=0 due to 1/U
#     max_k = {2: 2, 4: 3, 6: 4}.get(order, None)
#     if max_k is None:
#         raise ValueError("order must be one of {2, 4, 6}")
#     for s in range(1, max_k + 1):
#         if np.isclose(t0 + s*k*h, 0.0) or np.isclose(t0 - s*k*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 5-point, O(h^2):  (f_-2 - 2 f_-1 + 2 f_+1 - f_+2) / (2 h^3)
#         fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0- h)
#         fp1 = f(t0 - k*h, U0+h);   fp2 = f(t0 - 2*k*h, U0 + 2*h)
#         return (fm2 - 2*fm1 + 2*fp1 - fp2) / (2*h**3)

#     elif order == 4:
#         # 7-point, O(h^4): (f_-3 - 8 f_-2 + 13 f_-1 - 13 f_+1 + 8 f_+2 - f_+3) / (8 h^3)
#         fm3 = f(t0 + 3*k*h, U0 - 3*h); fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h); fp3 = f(t0 - 3*k*h, U0 + 3*h)
#         return (fm3 - 8*fm2 + 13*fm1 - 13*fp1 + 8*fp2 - fp3) / (8*h**3)


# def G3_FD_wce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order centered finite-difference for f'''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f'''(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     # Avoid evaluating at U=0 due to 1/U
#     max_k = {2: 2, 4: 3, 6: 4}.get(order, None)
#     if max_k is None:
#         raise ValueError("order must be one of {2, 4, 6}")
#     for s in range(1, max_k + 1):
#         if np.isclose(t0 + s*k*h, 0.0) or np.isclose(t0 - s*k*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 5-point, O(h^2):  (f_-2 - 2 f_-1 + 2 f_+1 - f_+2) / (2 h^3)
#         fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0+h);   fp2 = f(t0 - 2*k*h, U0 + 2*h)
#         return (fm2 - 2*fm1 + 2*fp1 - fp2) / (2*h**3)

#     elif order == 4:
#         # 7-point, O(h^4): (f_-3 - 8 f_-2 + 13 f_-1 - 13 f_+1 + 8 f_+2 - f_+3) / (8 h^3)
#         fm3 = f(t0 + 3*k*h, U0 - 3*h); fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h); fp3 = f(t0 - 3*k*h, U0 + 3*h)
#         return (fm3 - 8*fm2 + 13*fm1 - 13*fp1 + 8*fp2 - fp3) / (8*h**3)

# def G4_FD_wce(n, t0, U0, wn, k, h=1e-3, order=4):
#     """
#     High-order centered finite-difference for f''''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f''''(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     # Avoid division by zero
#     max_k = {2: 2, 4: 3, 6: 4}.get(order, None)
#     if max_k is None:
#         raise ValueError("order must be one of {2, 4, 6}")
#     for s in range(1, max_k+1):
#         if np.isclose(t0 + s*k*h, 0.0) or np.isclose(t0 - s*k*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 5-point, O(h^2)
#         fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0+h);   fp2 = f(t0 - 2*k*h, U0 + 2*h)
#         f0  = f(t0, U0)
#         return (fm2 - 4*fm1 + 6*f0 - 4*fp1 + fp2) / (h**4)

#     elif order == 4:
#         # 7-point, O(h^4)
#         fm3 = f(t0 + 3*k*h, U0 - 3*h); fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h); fp3 = f(t0 - 3*k*h, U0 + 3*h)
#         f0  = f(t0, U0)
#         num = (-fm3 + 12*fm2 - 39*fm1 + 56*f0 - 39*fp1 + 12*fp2 - fp3)
#         return num / (6*h**4)

# def G5_FD_wce(n, t0, U0, wn, k, h=1e-3, order=4):
#     """
#     High-order centered finite-difference for f^(5)(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f^(5)(t0).
#     """
#     def f(t,U):
#         return Gij_wce_series(n, t, U, wn)

#     # Central stencils (offsets, coefficients). Result is sum(c_i f(t0 + k_i h)) / h^5
#     stencils = {
#         # 7-point, O(h^2)
#         2: ([-3, -2, -1, 0, 1, 2, 3],
#             [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]),
#         # 9-point, O(h^4)
#         4: ([-4, -3, -2, -1, 0, 1, 2, 3, 4],
#             [ 1/6, -3/2, 13/3, -29/6, 0, 29/6, -13/3, 3/2, -1/6]),
#         # 11-point, O(h^6)
#         6: ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#             [-13/288, 19/36, -87/32, 13/2, -323/48, 0,
#               323/48, -13/2, 87/32, -19/36, 13/288]),
#     }
#     if order not in stencils:
#         raise ValueError("order must be one of {2, 4, 6}")

#     offs, coeffs = stencils[order]

#     # Avoid evaluating at U = 0 (f contains 1/U)
#     for s in offs:
#         if s != 0 and np.isclose(t0 + s*k*h, 0.0):
#             raise ZeroDivisionError("t0 ± s*k*h hits zero; adjust h or t0.")

#     vals = [f(t0 - s*k*h, U0 + s*h) for s in offs]
#     return sum(c*v for c, v in zip(coeffs, vals)) / (h**5)


# ############# wCE-Taylor expansion in terms of U-U0
# def Gij_FD_wce_series(n, wce_order, t, U, t0, U0, wn, k):
#     """
#     Exact series expansion for Green's function
    
#     Parameters:
#     - U (float): Interaction strength.
#     - wn (1D array): Matsubara frequencies.
    
#     Returns:
#     - Gn (2D array): Reconstructed Green's function [Nw* Ns* Ns].
#     """
#     if n > 5:
#         raise ValueError("Expansion order n is too large! Only n <=6 is supported.")
    
#     # Calculate the approximated Green's function
#     # print('t0 is', t0, 'U0 is', U0)
#     G0 = G0_FD_wce(wce_order, t0, U0, wn, k)
#     G1 = G1_FD_wce(wce_order, t0, U0, wn, k)
#     G2 = G2_FD_wce(wce_order, t0, U0, wn, k)/2
#     G3 = G3_FD_wce(wce_order, t0, U0, wn, k)/factorial(3)
#     G4 = G4_FD_wce(wce_order, t0, U0, wn, k)/factorial(4)
#     G5 = G5_FD_wce(wce_order, t0, U0, wn, k)/factorial(5)
#     if n == 0:
#         Gn = G0
#     elif n == 1:
#         Gn = G0+ (U-U0)*G1
#     elif n == 2:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2
#     elif n == 3:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3
#     elif n == 4:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3+(U-U0)**4*G4
#     elif n == 5:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3+(U-U0)**4*G4+(U-U0)**5*G5
#     return Gn


# def G0_FD_sce(n, t0, U0, wn, k):
#     G0_ij = Gij_sce_series(n, t0, U0, wn)
#     return G0_ij

# def G1_FD_sce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order central finite-difference for the first derivative f'(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float or complex
#     t0 : float
#     h : float, step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order of the scheme (default 4)

#     Returns
#     -------
#     complex
#         Approximation to f'(t0).
#     """
#     def f(t,U):
#         return Gij_sce_series(n, t, U, wn)

#     if order == 2:
#         # O(h^2) 3-point central
#         fp = f(t0 - k*h, U0 + h)
#         fm = f(t0 + k*h, U0 - h)
#         return (fp - fm) / (2*h)

#     elif order == 4:
#         # O(h^4) 5-point central
#         fp1 = f(t0 - k*h, U0 + h);   fm1 = f(t0 + k*h, U0-h)
#         fp2 = f(t0 - 2*k*h, U0 + 2*h); fm2 = f(t0 + 2*k*h, U0-2*h)
#         return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12*h)

#     elif order == 6:
#         # O(h^6) 7-point central
#         fp1 = f(t0 - k*h, U0 + h);   fm1 = f(t0 + k*h, U0 - h)
#         fp2 = f(t0 - 2*k*h, U0 + 2*h); fm2 = f(t0 + 2*k*h, U0-2*h)
#         fp3 = f(t0 - 3*k*h, U0 + 3*h); fm3 = f(t0 + 3*k*h, U0-3*h)
#         return (fm3 - 9*fm2 + 45*fm1 - 45*fp1 + 9*fp2 - fp3) / (60*h)

#     else:
#         raise ValueError("order must be one of {2, 4, 6}")


# def G2_FD_sce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order central finite-difference for the second derivative f''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f''(t0).
#     """
#     def f(t,U):
#         return Gij_sce_series(n, t, U, wn)

#     # Ensure we don't evaluate at U=0
#     for s in range(1, (order//2)*2):  # k=1,2 for order>=4; k=1..3 for order=6
#         if np.isclose(t0 + s*h, 0.0) or np.isclose(t0 - s*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 3-point central, O(h^2)
#         fp1 = f(t0 - k*h, U0 + h); f0 = f(t0, U0); fm1 = f(t0 + k*h, U0 - h)
#         return (fp1 - 2*f0 + fm1) / (h*h)

#     elif order == 4:
#         # 5-point central, O(h^4)
#         fp2 = f(t0 - 2*k*h, U0 + 2*h); fp1 = f(t0 - k*h, U0 + h)
#         f0  = f(t0, U0)
#         fm1 = f(t0 + k*h, U0 - h);   fm2 = f(t0 + 2*k*h, U0 - 2*h)
#         return (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*h*h)

#     elif order == 6:
#         # 7-point central, O(h^6)
#         fp3 = f(t0 - 3*k*h, U0 + 3*h); fp2 = f(t0 - 2*k*h, U0 + 2*h); fp1 = f(t0 - k*h, U0 + h)
#         f0  = f(t0, U0 )
#         fm1 = f(t0 + k*h, U0 - h);   fm2 = f(t0 + 2*k*h, U0 - 2*h); fm3 = f(t0 + 3*k*h, U0 - 3*h)
#         return (-fm3/90 + 3*fm2/20 - 3*fm1/2 + 49*f0/18
#                 - 3*fp1/2 + 3*fp2/20 - fp3/90) / (h*h)

#     else:
#         raise ValueError("order must be one of {2, 4, 6}")



# def G3_FD_sce(n, t0, U0, wn, k, h=1e-4, order=4):
#     """
#     High-order centered finite-difference for f'''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f'''(t0).
#     """
#     def f(t,U):
#         return Gij_sce_series(n, t, U, wn)

#     # Avoid evaluating at U=0 due to 1/U
#     max_k = {2: 2, 4: 3, 6: 4}.get(order, None)
#     if max_k is None:
#         raise ValueError("order must be one of {2, 4, 6}")
#     for s in range(1, max_k + 1):
#         if np.isclose(t0 + s*h, 0.0) or np.isclose(t0 - s*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 5-point, O(h^2):  (f_-2 - 2 f_-1 + 2 f_+1 - f_+2) / (2 h^3)
#         fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h)
#         return (fm2 - 2*fm1 + 2*fp1 - fp2) / (2*h**3)

#     elif order == 4:
#         # 7-point, O(h^4): (f_-3 - 8 f_-2 + 13 f_-1 - 13 f_+1 + 8 f_+2 - f_+3) / (8 h^3)
#         fm3 = f(t0 + 3*k*h, U0 - 3*h); fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h); fp3 = f(t0 - 3*k*h, U0 + 3*h)
#         return (fm3 - 8*fm2 + 13*fm1 - 13*fp1 + 8*fp2 - fp3) / (8*h**3)

#     # else:  # order == 6
#     #     # 9-point, O(h^6):
#     #     # (-7 f_-4 + 72 f_-3 - 338 f_-2 + 488 f_-1 - 488 f_+1 + 338 f_+2 - 72 f_+3 + 7 f_+4) / (240 h^3)
#     #     fm4 = f(t0 - 4*h); fm3 = f(t0 - 3*h); fm2 = f(t0 - 2*h); fm1 = f(t0 - h)
#     #     fp1 = f(t0 + h);   fp2 = f(t0 + 2*h); fp3 = f(t0 + 3*h); fp4 = f(t0 + 4*h)
#     #     num = (-7*fm4 + 72*fm3 - 338*fm2 + 488*fm1
#     #            - 488*fp1 + 338*fp2 - 72*fp3 + 7*fp4)
#         # return num / (240*h**3)


# def G4_FD_sce(n, t0, U0, wn, k, h=1e-3, order=4):
#     """
#     High-order centered finite-difference for f''''(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f''''(t0).
#     """
#     def f(t,U):
#         return Gij_sce_series(n, t, U, wn)

#     # Avoid division by zero
#     max_k = {2: 2, 4: 3, 6: 4}.get(order, None)
#     if max_k is None:
#         raise ValueError("order must be one of {2, 4, 6}")
#     for s in range(1, max_k+1):
#         if np.isclose(t0 + s*h, 0.0) or np.isclose(t0 - s*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     if order == 2:
#         # 5-point, O(h^2)
#         fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         f0  = f(t0, U0)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h)
#         return (fm2 - 4*fm1 + 6*f0 - 4*fp1 + fp2) / (h**4)

#     elif order == 4:
#         # 7-point, O(h^4)
#         fm3 = f(t0 + 3*k*h, U0 - 3*h); fm2 = f(t0 + 2*k*h, U0 - 2*h); fm1 = f(t0 + k*h, U0 - h)
#         fp1 = f(t0 - k*h, U0 + h);   fp2 = f(t0 - 2*k*h, U0 + 2*h); fp3 = f(t0 - 3*k*h, U0 + 3*h)
#         f0  = f(t0, U0)
#         num = (-fm3 + 12*fm2 - 39*fm1 + 56*f0 - 39*fp1 + 12*fp2 - fp3)
#         return num / (6*h**4)


# def G5_FD_sce(n, t0, U0, wn, k, h=1e-3, order=4):
#     """
#     High-order centered finite-difference for f^(5)(t0),
#     where f(U) = (1/U) * Gij_ana_series(n, 1/U, 1.0, wn/U).

#     Parameters
#     ----------
#     n : int
#     wn : float | complex | array-like
#     t0 : float
#     h  : float, base step size (default 1e-4)
#     order : {2, 4, 6}, accuracy order (default 4)

#     Returns
#     -------
#     complex or ndarray
#         Approximation to f^(5)(t0).
#     """
#     def f(t,U):
#         return Gij_sce_series(n, t, U, wn)

#     # Central stencils (offsets, coefficients). Result is sum(c_i f(t0 + k_i h)) / h^5
#     stencils = {
#         # 7-point, O(h^2)
#         2: ([-3, -2, -1, 0, 1, 2, 3],
#             [-1/2, 2, -5/2, 0, 5/2, -2, 1/2]),
#         # 9-point, O(h^4)
#         4: ([-4, -3, -2, -1, 0, 1, 2, 3, 4],
#             [ 1/6, -3/2, 13/3, -29/6, 0, 29/6, -13/3, 3/2, -1/6]),
#         # 11-point, O(h^6)
#         6: ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#             [-13/288, 19/36, -87/32, 13/2, -323/48, 0,
#               323/48, -13/2, 87/32, -19/36, 13/288]),
#     }
#     if order not in stencils:
#         raise ValueError("order must be one of {2, 4, 6}")

#     offs, coeffs = stencils[order]

#     # Avoid evaluating at U = 0 (f contains 1/U)
#     for s in offs:
#         if s != 0 and np.isclose(t0 + s*k*h, 0.0):
#             raise ZeroDivisionError("t0 ± k*h hits zero; adjust h or t0.")

#     vals = [f(t0 - s*k*h, U0 +s*h) for s in offs]
#     return sum(c*v for c, v in zip(coeffs, vals)) / (h**5)

    
# ############# SCE-Taylor expansion in terms of U-U0
# def Gij_FD_sce_series(n, sce_order, t, U, t0, U0, wn, k):
#     """
#     Exact series expansion for Green's function
    
#     Parameters:
#     - U (float): Interaction strength.
#     - wn (1D array): Matsubara frequencies.
    
#     Returns:
#     - Gn (2D array): Reconstructed Green's function [Nw* Ns* Ns].
#     """
#     if n > 5:
#         raise ValueError("Expansion order n is too large! Only n <=5 is supported.")

#     G0 = G0_FD_sce(sce_order, t0, U0, wn, k)
#     G1 = G1_FD_sce(sce_order, t0, U0, wn, k)
#     G2 = G2_FD_sce(sce_order, t0, U0, wn, k)/2
#     G3 = G3_FD_sce(sce_order, t0, U0, wn, k)/factorial(3)
#     G4 = G4_FD_sce(sce_order, t0, U0, wn, k)/factorial(4)
#     G5 = G5_FD_sce(sce_order, t0, U0, wn, k)/factorial(5)
    
#     if n == 0:
#         Gn = G0
#     elif n == 1:
#         Gn = G0+ (U-U0)*G1
#     elif n == 2:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2
#     elif n == 3:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3
#     elif n == 4:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3+(U-U0)**4*G4
#     elif n == 5:
#         Gn = G0 +(U-U0)*G1+ (U-U0)**2*G2+(U-U0)**3*G3+(U-U0)**4*G4+(U-U0)**5*G5
#     return Gn

