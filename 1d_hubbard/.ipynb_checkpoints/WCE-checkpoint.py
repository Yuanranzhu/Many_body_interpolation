### This gathers the utilities function for pade approximation for Green's function of 2D hubbard model 
import itertools as it
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pydlr import dlr
################################### pydrl 
def Gij_to_Gk(G):
    """
    Usage 1:
    Using FFT to: G_ij(tau) --> G(k,tau)
    
    Parameters:
    - G: array of Ntau * Ns * Ns 
    - Ntau: int, number of time point in [0,beta]
    - Ns: int, number of (periodic)systems site
    
    Returns:
    - Gk: array of Nk* Ntau, Nk=Ns
    
    Usage 2:
    Using FFT to: G_ij(iwn) --> G(k,iwn)
    
    Parameters:
    - G: array of Nw * Ns * Ns 
    - Nw: int, number of Matsubara frequencies
    - Ns: int, number of (periodic)systems site
    
    Returns:
    - Gk: array of Nk* Nw, Nk=Ns
    """
    ## Using FFT to get k-space Green's function (periodic system)
    N = G.shape[0]
    Ns = G.shape[1]
    Gk = np.zeros((Ns,N),dtype='complex')
    if Ns%4 == 0:
        kd = np.linspace(0,Ns-1,Ns)
    else:
        kd = np.linspace(-Ns+1,Ns-1,Ns)/2
    for k in np.arange(Ns):
        for j in np.arange(Ns):
            Gk[k,:] += G[:,0,j]*np.exp(-2*np.pi*1j*kd[k]*j/Ns) 
    return Gk/Ns


def Gk_to_Gij(Gk):
    """
    Usage 1: 
    Using IFFT to : G(k,wn) --> G_ij(wn)
    
    Parameters:
    - Gk: array of Nk * Nw
    - Nw: int, number of Matsubara frequencies
    - Nk: int, number of k points in the (periodic)systems
    
    Returns:
    - G_{ij}(iw_n): array of Nw * Ns * Ns, Ns = Nk
    
    Usage 2:
    Using IFFT to : G(k,tau) --> G_ij(tau)
    
    Parameters:
    - Gk: array of Nk * Ntau
    - Ntau: int, number of time point in [0,beta]
    - Nk: int, number of k points in the (periodic)systems
    
    Returns:
    - G_{ij}(tau): array of Ntau * Ns * Ns, Ns = Nk
    
    """
    
    ## Using IFFT to get lattice Green's function (periodic system)
    Ns = Gk.shape[0]
    N = Gk.shape[1]
    G = np.zeros((N,Ns,Ns),dtype='complex')
    
    if Ns%4 == 0:
        kd = np.linspace(0,Ns-1,Ns)
    else:
        kd = np.linspace(-Ns+1,Ns-1,Ns)/2
        
    for j in np.arange(Ns):
        for k in np.arange(kd.size):
            G[:,0,j] += Gk[k,:]*np.exp(2*np.pi*1j*kd[k]*j/Ns)
        G[:,j,0] = G[:,0,j]
        for i in np.arange(Ns):
            if i+j < Ns:
                G[:,i,i+j] = G[:,0,j]
                G[:,i+j,i] = G[:,0,j]   
    return G
    

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

################################### WCE
def HF_G0(t, Ns, wn):
    """
    Solving dyson's eqn to get the HF Green's function
    
    Input: H0 (Ns*Ns), wn(vector of Matsubabra frequencies)
    
    Returns:
    - G0: array of Nw*Ns*Ns
    - G0k_iwn: array of Ns*Nw
    """
    
    G0 = np.zeros((wn.size,Ns,Ns),dtype=complex)
    h0 = np.zeros((Ns,Ns))
    for i in range(Ns):
        if i + 1 < Ns:  
            h0[i, i + 1] = -1.0 
        if i - 1 >= 0: 
            h0[i, i - 1] = -1.0
    if Ns%4 ==0:
        h0[0,-1] = -1.0 
        h0[-1,0] = -1.0     
    for i in np.arange(wn.size):
        G0[i,:,:] = np.linalg.inv(1j*wn[i]*np.eye(Ns, dtype=complex)+ t*h0)
        
    
    if Ns%4 ==0:
        ck = np.cos(np.linspace(-Ns/2,Ns/2,Ns+1)*2*np.pi/Ns)
        ck = ck[:-1]
    else:
        ck = -np.cos(np.pi*np.linspace(-Ns+1,Ns-1,Ns)/Ns)
        
    G0k_iwn =  np.zeros((Ns,wn.size),dtype='complex')
    for k in np.arange(Ns):
        for n in np.arange(wn.size):
            G0k_iwn[k,n] = 1/(1j*wn[n]+2*t*ck[k])
            
    return G0, G0k_iwn



def sigma_2ndB(G0k, wn, beta):
    """
    Calculating the bare 2ndB self-energy ## Test this function 
    """
    Nk, Nw = G0k.shape
    tau = np.linspace(0,beta, 401)
    G0ij_tau = Gij_iwn_to_Gij_tau(Gk_to_Gij(-G0k),wn, tau, beta)
    G0ji_tau = np.conj(G0ij_tau).transpose(0, 2, 1)
    Sij_tau = np.zeros((tau.size, Nk,Nk), dtype='complex')
    for i in range(tau.size): 
        Sij_tau[i,:,:] = -G0ij_tau[i,:,:]*G0ij_tau[i,:,:]*G0ji_tau[-i,:,:]
    Sk_iwn = Gij_to_Gk(Gij_tau_to_Gij_iwn(Sij_tau, wn, tau, beta))
    return Gij_to_Gk(Sij_tau), Sk_iwn


def Gij_wce_U(n, Nk, U, wn, beta, t):
    """
    Weak coupling expansion for Green's function using the bare self-energy 
    
    Parameters:
    - U (float): Interaction strength.
    - wn (1D array): Matsubara frequencies.
    - n (int): Expansion order (currently supports n <= 2).
    
    Returns:
    - Gn (2D array): Reconstructed Green's function [Nk* Nw].
    """
    if n > 2:
        raise ValueError("Expansion order n is too large! Only n <= 2 is supported.")
    
    # # Calculate the WCE 
    _, a0k_iwn = HF_G0(t, Nk, wn)
    _, Sk_iwn = sigma_2ndB(a0k_iwn/ Nk, wn, beta)  # Self-energy in k-iwn space
    a2k_iwn = a0k_iwn* Sk_iwn *a0k_iwn
    
    a0_ij = Gk_to_Gij(a0k_iwn/Nk)
    a2_ij = Gk_to_Gij(a2k_iwn) 
    if n == 1:
        Gij_iwn =  a0_ij
    elif n ==2:
        Gij_iwn = a0_ij + U**2*a2_ij
    
    return Gij_iwn 