### This gathers the utilities function for pade approximation for Green's function of 2D hubbard model 
import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pydlr import dlr
import utilities_2d as ut_2d


def HF_G0(Nkx, Nky, wn, t):
    """
    Solving Dyson's equation to get the Hartree-Fock Green's function.

    Parameters:
    Nkx (int): Number of kx points.
    Nky (int): Number of ky points.
    wn (np.array): Vector of Matsubara frequencies.

    Returns:
    - G0: array of shape (Nw, Nkx, Nky)
    """
    G0k = np.zeros((wn.size, Nkx, Nky), dtype=complex)

    kx = 2 * np.pi * np.arange(Nkx) / Nkx
    ky = 2 * np.pi * np.arange(Nky) / Nky

    for i in range(wn.size):
        for j in range(Nkx):
            for k in range(Nky):
                if Nkx == 2:
                    epsilon_k = -(np.cos(kx[j]) + np.cos(ky[k]))  # Dispersion relation
                else:
                    epsilon_k = -2 * (np.cos(kx[j]) + np.cos(ky[k]))  # Dispersion relation
                G0k[i, j, k] = 1.0 / (1j * wn[i] - t*epsilon_k)  # Dyson's equation
    G0ij = ut_2d.Gkxky_to_Gij(G0k, Nkx, Nky)
    return G0ij, G0k

def sigma_2ndB(Gk, wn, beta, E_max=10.0, err=1e-10):
    """
    Calculating the bare 2ndB self-energy ## Test this function
    Input: 
    - G[kx, ky, iwn]: array of (Nw * Nkx *Nky) 
    - wn(vector of Matsubabra frequencies)
    
    Returns:
    - S[kx, ky, iwn]: array of (Nw * Nkx *Nky) 
    """
    Nw, Nkx, Nky= Gk.shape
    tau = np.linspace(0,beta, 401)
    Gij_iwn = ut_2d.Gkxky_to_Gij(Gk, Nkx, Nky)
    d = dlr(lamb=beta*E_max, eps= err)
    Gij_dlr = d.lstsq_dlr_from_matsubara(1j*wn, Gij_iwn, beta)
    Gij_tau = d.eval_dlr_tau(Gij_dlr, tau, beta)
    Gji_tau = np.conj(Gij_tau).transpose(0, 2, 1)
    Sij_tau = np.zeros((tau.size,Nkx*Nky,Nkx*Nky), dtype='complex')
    for i in range(tau.size): 
        Sij_tau[i,:,:] = Gij_tau[i,:,:]*Gij_tau[i,:,:]*Gji_tau[-i,:,:]
    
    Sij_dlr  = d.lstsq_dlr_from_tau(tau, Sij_tau, beta)
    Sij_iwn = d.eval_dlr_freq(Sij_dlr, 1j*wn, beta)
    Skxky_iwn = ut_2d.Gij_to_Gkxky(Sij_iwn, Nkx, Nky) 
    return Skxky_iwn




def dyson_G_kiwn(G0k, Sk):
    """
    Solving Dyson's eqn to get the G
    
    Input: 
    - S[k,iwn] (Nw * Nkx * Nky), wn(vector of Matsubabra frequencies)
    - G0[k,iwn] (Nw * Nkx * Nky), wn(vector of Matsubabra frequencies)
    
    Returns:
    - G[k,iwn] (Nw * Mkx * Nky), wn(vector of Matsubabra frequencies)
    """
    Nw, Nkx, Nky = G0k.shape
    Gk = np.zeros((Nw, Nkx, Nky), dtype='complex')
    for i in range(Nkx):
        for j in range(Nky):
            for k in range(Nw):
                Gk[k,i,j] = (G0k[k,i,j]**(-1) - Sk[k,i,j])**(-1)
    return Gk
    
    
def G_wce(n, Nkx, Nky, U, t, wn, beta):
    """
    Weak coupling expansion for Green's function using the bare self-energy 
    
    Parameters:
    - U (float): Interaction strength.
    - wn (1D array): Matsubara frequencies.
    - n (int): Expansion order (currently supports n <= 2).
    
    Returns:
    - Gn (2D array): Reconstructed Green's function [Nw*Nkx* Nky].
    """
    if n > 2:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")
    
    _, G0k = HF_G0(Nkx, Nky, wn, t)
    Sk = sigma_2ndB(G0k, wn, beta)    # 2ndB Self-energy in k, iwn space
    
    # Initialize terms
    second_term = 0
    third_term = U**2*G0k* Sk *G0k
    
    # Calculate the approximated Green's function
    if n == 0:
        Gk = G0k
    elif n == 1:
        Gk = G0k + second_term 
    elif n == 2:
        Gk = G0k + second_term + third_term
    Gij = ut_2d.Gkxky_to_Gij(Gk, Nkx, Nky)
    return Gij, Gk 