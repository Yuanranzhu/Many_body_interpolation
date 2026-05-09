### This file contains all utility functions
import itertools as it
import pickle
import numpy as np
from pydlr import dlr
import cvxpy as cp

def G_vec_to_mat(G_real,G_imag,Ns,Ntau):
    """ 
    Loading the MBPT Green's function or self-energy
    
    Returns:
    - G: array of Ns* Ns * Ntau

    """
    G = np.zeros((Ns,Ns,Ntau),dtype="complex")
    for i in np.arange(Ntau):
        G[:,:,i] = G_real[i*Ns:(i+1)*Ns,:] + 1j*G_imag[i*Ns:(i+1)*Ns,:]
    return G 

def G_lattice_interp(G_lattice, beta, Ntau, E_max=10.0, eps=1e-10):
    """
    Using pydlr to interpolate the Green's function or self-energy from time-domain [0,beta-dt] to [0,beta]
    
    Parameters: 
    - G_lattice: array of Ns*Ns*Ntau
    - default parameters of the pydlr solver
    
    Returns:
    - G_lattice_interpolate: array of Ntau*Ns*Ns
    
    """
    G_lattice_interpolate = np.transpose(G_lattice,(2,0,1))
    return G_lattice_interpolate
    

def Gij_iwn_to_Gij_tau(G, wn, tau, beta, E_max=10.0, err=1e-10):
    """
    Using pydlr to: G(ij,iw_n) --> G(ij,tau)
    """
    d = dlr(lamb=beta*E_max, eps= err)
    Nw, Ns, _= G.shape
    Gij_tau = np.zeros((Ns,tau.size))
    Gij_dlr = d.lstsq_dlr_from_matsubara(1j*wn, G, beta)
    Gij_tau_c = d.eval_dlr_tau(Gij_dlr, tau, beta)
    Gij_tau = np.real(Gij_tau_c)
    return Gij_tau


def Gij_tau_to_Gij_iwn(G, wn, tau, beta, E_max=10.0, err=1e-10):
    """
    Using pydlr to: G(ij,iw_n) --> G(ij,tau)
    """
    d = dlr(lamb=beta*E_max, eps= err)
    Ntau, Ns, _= G.shape
    Gij_dlr = d.lstsq_dlr_from_tau(tau, G, beta)
    Gij_iwn = d.eval_dlr_freq(Gij_dlr, 1j*wn, beta)
    return Gij_iwn

def Gij_to_Gkxky(G_real, Lx, Ly):
    """
    Compute the Fourier transform of G_ij(tau) to G(k_x, k_y, tau).
    
    Parameters:
    G_real : np.array of shape (N_tau, Lx*Ly, Lx*Ly)
        The Matsubara Green's function in real space.
    Lx, Ly : int
        Lattice dimensions.
    
    Returns:
    G_k : np.array of shape (N_tau, Lx, Ly)
        The Green's function in momentum space.
    """
    N_tau = G_real.shape[0]  # Number of imaginary time points
    G_k = np.zeros((N_tau, Lx, Ly), dtype=complex)

    # Generate real-space lattice coordinates
    coords = np.array([(i // Ly, i % Ly) for i in range(Lx * Ly)])

    # Compute Fourier transform for each tau index
    for tau in range(N_tau):
        G_tau = G_real[tau]  # (Lx*Ly, Lx*Ly)
        for kx_idx, kx in enumerate(np.fft.fftfreq(Lx) * 2 * np.pi):
            for ky_idx, ky in enumerate(np.fft.fftfreq(Ly) * 2 * np.pi):
                phase_factor = np.exp(-1j * (kx * (coords[:, 0][:, None] - coords[:, 0][None, :]) +
                                             ky * (coords[:, 1][:, None] - coords[:, 1][None, :])))
                G_k[tau, kx_idx, ky_idx] = np.sum(phase_factor * G_tau)

    return G_k/(Lx*Ly)

    
def Gkxky_to_Gij(G_k, Lx, Ly):
    """
    Compute the inverse Fourier transform of G(k_x, k_y, tau) to G_ij(tau).
    
    Parameters:
    G_k : np.array of shape (N_tau, Lx, Ly)
        The Matsubara Green's function in momentum space.
    Lx, Ly : int
        Lattice dimensions.
    
    Returns:
    G_real : np.array of shape (N_tau, Lx*Ly, Lx*Ly)
        The Green's function in real space.
    """
    N_tau = G_k.shape[0]  # Number of imaginary time points
    G_real = np.zeros((N_tau, Lx * Ly, Lx * Ly), dtype=complex)

    # Generate real-space lattice coordinates
    coords = np.array([(i // Ly, i % Ly) for i in range(Lx * Ly)])

    # Compute inverse Fourier transform for each tau index
    for tau in range(N_tau):
        G_tau_k = G_k[tau]  # (Lx, Ly)
        for i, (xi, yi) in enumerate(coords):
            for j, (xj, yj) in enumerate(coords):
                phase_factor = np.exp(1j * (np.fft.fftfreq(Lx) * 2 * np.pi * (xi - xj))[:, None] +
                                      1j * (np.fft.fftfreq(Ly) * 2 * np.pi * (yi - yj))[None, :])
                G_real[tau, i, j] = np.sum(phase_factor * G_tau_k) 

    return G_real/(Lx*Ly)





