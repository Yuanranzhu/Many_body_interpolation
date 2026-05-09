import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt

def G_analytic_old(U, wn, t = 1.0):
    """
    Compute the 2x2 matrix G_{ij} for i,j = 1,2.

    Parameters:
    - a: float
    - c: float
    - iw_n: complex or array-like (Matsubara frequency i*omega_n)
    - t: float, hopping parameter (default = 1.0)
    - U: float, interaction strength (default = 1.0)

    Returns:
    - G: 2x2 complex-valued numpy array
    """
    iw_n = 1j*wn
    c = np.sqrt(16*t**2+ U**2)
    a = np.sqrt(32*t**2+2*(c-U)**2)/ (c-U)
    G = np.zeros((wn.size, 2, 2), dtype=complex)
    
    if np.abs(t)<1e-8:
        G[:, 0, 0] = iw_n/((iw_n)**2-U**2/4)
        G[:, 1, 1] = iw_n/((iw_n)**2-U**2/4)
        return G
        
    # Precompute terms
    alpha = 4 * t / (c - U)
    alpha_plus_sq = (1 + alpha)**2
    alpha_minus_sq = (1 - alpha)**2


    denom1 = iw_n + t - c / 2
    denom2 = iw_n - t - c / 2
    denom3 = iw_n - t + c / 2
    denom4 = iw_n + t + c / 2

    for i in range(2):
        for j in range(2):
            sign = (-1)**(i - j)
            G[:, i, j] = (sign / (2 * a**2) * (alpha_plus_sq / denom1 +sign * alpha_minus_sq / denom2) +
                1 / (2 * a**2) * (
                    alpha_plus_sq / denom3 +
                    sign * alpha_minus_sq / denom4
                )
            )
    return G

def G_analytic(t, wn, U, beta):
    """
    Compute the 2x2 matrix G_{ij} for i,j = 1,2.

    Parameters:
    - a: float
    - c: float
    - iw_n: complex or array-like (Matsubara frequency i*omega_n)
    - t: float, hopping parameter (default = 1.0)
    - U: float, interaction strength (default = 1.0)

    Returns:
    - G: 2x2 complex-valued numpy array
    """
    iw_n = 1j*wn
    G = np.zeros((wn.size, 2, 2), dtype=complex)
    
    if np.abs(t)<1e-8:
        G[:, 0, 0] = iw_n/((iw_n)**2-U**2/4)
        G[:, 1, 1] = iw_n/((iw_n)**2-U**2/4)
        return G
        
    c = np.sqrt(16*t**2+ U**2)
    a = np.sqrt(32*t**2+2*(c-U)**2)/ (c-U)

    
    # Precompute terms
    alpha = 4 * t / (c - U)
    alpha_plus_sq = (1 + alpha)**2
    alpha_minus_sq = (1 - alpha)**2



    # Z = 1+ 3*np.exp(beta*(U/2-c/2)) + 2*np.exp(-beta*(U/2+c/2)) + 4*np.exp(beta*(t-c/2)) + 4*np.exp(beta*(-t-c/2)) + np.exp(-c)
    Z = 1+ 3*np.exp(beta*(U/2-c/2)) 
    factor1 = (1+np.exp(beta*(t-c/2)))/Z
    factor2 = (1+np.exp(beta*(-t-c/2)))/Z
    factor3 = (np.exp(beta*(t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    factor4 = (np.exp(beta*(-t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    # print('\n t is', t)
    # print('\n factor 1 is:', factor1, 'factor 2 is:', factor2)
    
    denom1 = iw_n + t - c / 2
    denom2 = iw_n - t - c / 2
    denom3 = iw_n - t + c / 2
    denom4 = iw_n + t + c / 2

    fdenom1 = iw_n + t - U / 2
    fdenom2 = iw_n - t - U / 2
    fdenom3 = iw_n - t + U / 2
    fdenom4 = iw_n + t + U / 2

    for i in range(2):
        for j in range(2):
            sign = (-1)**(i - j)
            G[:, i, j] = factor1/ (2 * a**2) * (sign* alpha_plus_sq / denom1 + alpha_minus_sq / denom2) + factor2/ (2 * a**2) * (alpha_plus_sq / denom3 +sign * alpha_minus_sq / denom4)+ factor3*3/4*(1/fdenom1+sign/fdenom3) +factor4*3/4*(sign/fdenom2+1/fdenom4) 
    return G


def G_analytic_tapp0(t, wn, U, beta):
    """
    Compute the 2x2 matrix G_{ij} for i,j = 1,2.

    Parameters:
    - a: float
    - c: float
    - iw_n: complex or array-like (Matsubara frequency i*omega_n)
    - t: float, hopping parameter (default = 1.0)
    - U: float, interaction strength (default = 1.0)

    Returns:
    - G: 2x2 complex-valued numpy array
    """
    iw_n = 1j*wn
    G = np.zeros((wn.size, 2, 2), dtype=complex)
    
    if np.abs(t)<1e-8:
        G[:, 0, 0] = iw_n/((iw_n)**2-U**2/4)
        G[:, 1, 1] = iw_n/((iw_n)**2-U**2/4)
        return G
        
    c = np.sqrt(16*t**2+ U**2)
    a = np.sqrt(32*t**2+2*(c-U)**2)/ (c-U)

    factor1 = 1/4
    factor2 = 1/4
    # print('\n t is', t)
    # print('\n factor 1 is:', factor1, 'factor 2 is:', factor2)
    
    denom1 = iw_n + t - c / 2
    denom2 = iw_n - t - c / 2
    denom3 = iw_n - t + c / 2
    denom4 = iw_n + t + c / 2

    fdenom1 = iw_n + t - U / 2
    fdenom2 = iw_n - t - U / 2
    fdenom3 = iw_n - t + U / 2
    fdenom4 = iw_n + t + U / 2

    for i in range(2):
        for j in range(2):
            sign = (-1)**(i - j)
            G[:, i, j] = factor1/ (2 * a**2) * (sign* alpha_plus_sq / denom1 + alpha_minus_sq / denom2) + factor1/ (2 * a**2) * (alpha_plus_sq / denom3 +sign * alpha_minus_sq / denom4)+ factor2*3/4*(1/fdenom1+sign/fdenom2+sign/fdenom3+1/fdenom4)
    return G



def G00_analytic(t, wn, U, beta):
    """
    Compute G_{00}.

    Parameters:
    - a: float
    - c: float
    - iw_n: complex or array-like (Matsubara frequency i*omega_n)
    - t: float, hopping parameter (default = 1.0)
    - U: float, interaction strength (default = 1.0)

    Returns:
    - G: 1d complex-valued numpy array
    """
    iw_n = 1j*wn
    
        
    c = np.sqrt(16*t**2+ U**2)
    a = np.sqrt(32*t**2+2*(c-U)**2)/ (c-U)

    
    # Precompute terms
    alpha = 4 * t / (c - U)
    alpha_plus_sq = (1 + alpha)**2
    alpha_minus_sq = (1 - alpha)**2



    # Z = 1+ 3*np.exp(beta*(U/2-c/2)) + 2*np.exp(-beta*(U/2+c/2)) + 4*np.exp(beta*(t-c/2)) + 4*np.exp(beta*(-t-c/2)) + np.exp(-c)
    Z = 1+ 3*np.exp(beta*(U/2-c/2)) 
    factor1 = (1+np.exp(beta*(t-c/2)))/Z
    factor2 = (1+np.exp(beta*(-t-c/2)))/Z
    factor3 = (np.exp(beta*(t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    factor4 = (np.exp(beta*(-t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    # print('\n t is', t)
    # print('\n factor 1 is:', factor1, 'factor 2 is:', factor2)
    
    denom1 = iw_n + t - c / 2
    denom2 = iw_n - t - c / 2
    denom3 = iw_n - t + c / 2
    denom4 = iw_n + t + c / 2

    fdenom1 = iw_n + t - U / 2
    fdenom2 = iw_n - t - U / 2
    fdenom3 = iw_n - t + U / 2
    fdenom4 = iw_n + t + U / 2


    sign = 1
    G = factor1/ (2 * a**2) * (sign* alpha_plus_sq / denom1 + alpha_minus_sq / denom2) + factor2/ (2 * a**2) * (alpha_plus_sq / denom3 +sign * alpha_minus_sq / denom4)+ factor3*3/4*(1/fdenom1+sign/fdenom3) +factor4*3/4*(sign/fdenom2+1/fdenom4) 
    return G


def G01_analytic(t, wn, U, beta):
    """
    Compute the G00

    Parameters:
    - a: float
    - c: float
    - iw_n: complex or array-like (Matsubara frequency i*omega_n)
    - t: float, hopping parameter (default = 1.0)
    - U: float, interaction strength (default = 1.0)

    Returns:
    - G: 1d complex-valued numpy array
    """
    iw_n = 1j*wn

        
    c = np.sqrt(16*t**2+ U**2)
    a = np.sqrt(32*t**2+2*(c-U)**2)/ (c-U)

    
    # Precompute terms
    alpha = 4 * t / (c - U)
    alpha_plus_sq = (1 + alpha)**2
    alpha_minus_sq = (1 - alpha)**2



    # Z = 1+ 3*np.exp(beta*(U/2-c/2)) + 2*np.exp(-beta*(U/2+c/2)) + 4*np.exp(beta*(t-c/2)) + 4*np.exp(beta*(-t-c/2)) + np.exp(-c)
    Z = 1+ 3*np.exp(beta*(U/2-c/2)) 
    factor1 = (1+np.exp(beta*(t-c/2)))/Z
    factor2 = (1+np.exp(beta*(-t-c/2)))/Z
    factor3 = (np.exp(beta*(t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    factor4 = (np.exp(beta*(-t-c/2))+np.exp(beta*(U/2-c/2)))/Z
    # print('\n t is', t)
    # print('\n factor 1 is:', factor1, 'factor 2 is:', factor2)
    
    denom1 = iw_n + t - c / 2
    denom2 = iw_n - t - c / 2
    denom3 = iw_n - t + c / 2
    denom4 = iw_n + t + c / 2

    fdenom1 = iw_n + t - U / 2
    fdenom2 = iw_n - t - U / 2
    fdenom3 = iw_n - t + U / 2
    fdenom4 = iw_n + t + U / 2

 
    sign = -1
    G = factor1/ (2 * a**2) * (sign* alpha_plus_sq / denom1 + alpha_minus_sq / denom2) + factor2/ (2 * a**2) * (alpha_plus_sq / denom3 +sign * alpha_minus_sq / denom4)+ factor3*3/4*(1/fdenom1+sign/fdenom3) +factor4*3/4*(sign/fdenom2+1/fdenom4) 
    return G
