### This file contains all utility functions
import itertools as it
import pickle
import numpy as np
from pydlr import dlr
import cvxpy as cp


def make_kernel(tau, wgrid, beta):
    K = np.zeros((len(tau),len(wgrid)))
    for i in range(len(tau)):
        for j in range(len(wgrid)):
            K[i,j] = np.exp(-tau[i]*wgrid[j])/(1+np.exp(-beta*wgrid[j]))
    return K 


def AC(G, tau, wgrid, beta, mu=1):
    """
    Analytical continuation of G_ii(tau) to get the spectral function
    
    Input:
    - G: array of 1*Ntau
    - tau: array of 1*Ntau
    - wgrid: array of 1*Nw
    Output: 
    - rho: arary of Nw 
    """
    kernel = make_kernel(tau, wgrid, beta)
    rho = cp.Variable(len(wgrid))
    constraints = [rho >= 0]
    cost = cp.sum_squares(kernel@rho + G) + mu*cp.sum_squares(rho[2:] - 2*rho[1:-1] +rho[:-2])
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    return rho.value

    
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

def load_file(filename):
    data = []
    with open(filename, 'r') as file:
        # Skip the first three lines containing metadata
        for _ in range(3):
            file.readline()
        
        # Read the data
        for line in file:
            if not line.startswith('#'):
                values = list(map(float, line.strip().split()))
                data.append(values)

    # Convert the data to a NumPy array for easier manipulation
    data = np.array(data)
    
    return data


def G_lattice_interp(G_lattice, beta, Ntau, E_max=10.0, eps=1e-10):
    return np.transpose(G_lattice,(2,0,1))
    
def Edlib_to_G(G):
    """ 
    Loading the ED Green's function or self-energy data
    
    Returns:
    - mat_fre: array of Nw*1 (Matsubara frequencies)
    - G_mat: array of Nw*Ns*Ns (Lattice G/S in Mat freq space)
    
    """
    mat_fre = G[:,0]
    Gv = G[:,1:]
    Re_Gv = Gv[:,::4]
    Im_Gv = Gv[:,1::4]
    Nw, Ns2 = Re_Gv.shape
    Ns = int(np.sqrt(Ns2))
    G_mat = Re_Gv+1j*Im_Gv
    G_mat = G_mat.reshape(Nw, Ns, Ns)
    return mat_fre, G_mat


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


def load_file(filename):
    data = []
    with open(filename, 'r') as file:
        # Skip the first three lines containing metadata
        for _ in range(3):
            file.readline()
        
        # Read the data
        for line in file:
            if not line.startswith('#'):
                values = list(map(float, line.strip().split()))
                data.append(values)

    # Convert the data to a NumPy array for easier manipulation
    data = np.array(data)
    
    return data


def Saving_data_ED(Ns, t_values, U_values, Ntau, data_loading_directory, data_saving_directory, full_sigma, beta):
    
    """
    Saves ED and data for different N_k and different U. 
    The whole data is saved into a tensor list.
    
    Parameters:
    - Ns_values: array-like, list of Ns values
    - U_values: array-like, list of U values
    - Ntau: int, number of tau points
    - data_directory: str, directory where ED data files are located
    - full_sigma: str, "True" for full self-energy or "False" for bare self-energy
    
    Returns (autosave):
    - G_data, Interactive G (encoded) : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    - G0_data, Bare G (encoded) : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    - S_data, Self-energy : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    """


    Gij_iwn_data = np.zeros((len(U_values), 800, Ns, Ns), dtype='complex')   #Gij_iwn
    Gij_tau_data = np.zeros((len(U_values), Ntau, Ns, Ns))   #Gij_tau
    Gk_tau_data = np.zeros((len(U_values), Ns, Ntau))   #Gk_tau
    Gk_iwn_data = np.zeros((len(U_values), Ns, 800), dtype='complex')   #Gk_iwn
        
    for i in np.arange(len(U_values)):
        G_data = load_file(data_loading_directory+ 'G_ij_omega_Ns%d_U%.1f_t%.1f_beta%.1f'%(Ns, U_values[i], t_values[i], beta))
        Mat_fre, G = Edlib_to_G(G_data)
        wn = np.append(-np.flip(Mat_fre),Mat_fre)
        tau = np.linspace(0, beta, Ntau)
        neg_G = np.flip(G,axis=0)
        Gij_iwn = np.append(np.transpose(neg_G.conj(),axes=(0, 2, 1)),G[:],axis=0)
            
        Gij_iwn_data[i] = Gij_iwn
        Gij_tau_data[i] = Gij_iwn_to_Gij_tau(Gij_iwn, wn, tau, beta)
        Gk_tau_data[i] = np.real(Gij_to_Gk(Gij_tau_data[i]))
        Gk_iwn_data[i] = Gij_to_Gk(Gij_iwn)

        
    # Save data using pickle to handle lists of arrays with different shapes
    data_filename = f'{data_saving_directory}/ED_Ns{Ns}_beta{beta}.pkl'
    with open(data_filename, 'wb') as f:
        pickle.dump({'Gij_tau_data': Gij_tau_data, 'Gij_iwn_data': Gij_iwn_data, 'Gk_tau_data': Gk_tau_data, 'Gk_iwn_data': Gk_iwn_data}, f)
    return None

def Saving_data_ED_varU(Ns, t_values, U_values, Ntau, data_loading_directory, data_saving_directory, full_sigma, beta):
    
    """
    Saves ED and data for different N_k and different U. 
    The whole data is saved into a tensor list.
    
    Parameters:
    - Ns_values: array-like, list of Ns values
    - U_values: array-like, list of U values
    - Ntau: int, number of tau points
    - data_directory: str, directory where ED data files are located
    - full_sigma: str, "True" for full self-energy or "False" for bare self-energy
    
    Returns (autosave):
    - G_data, Interactive G (encoded) : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    - G0_data, Bare G (encoded) : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    - S_data, Self-energy : List of tensors, Len(List) = #Nk, Each tensor: #U*Nk*(Ntau)
    """


    Gij_iwn_data = np.zeros((len(U_values), 800, Ns, Ns), dtype='complex')   #Gij_iwn
    Gij_tau_data = np.zeros((len(U_values), Ntau, Ns, Ns))   #Gij_tau
    Gk_tau_data = np.zeros((len(U_values), Ns, Ntau))   #Gk_tau
    Gk_iwn_data = np.zeros((len(U_values), Ns, 800), dtype='complex')   #Gk_iwn
        
    for i in np.arange(len(U_values)):
        G_data = load_file(data_loading_directory+ 'G_ij_omega_Ns%d_U%.1f_t%.1f_beta%.1f'%(Ns, U_values[i], t_values[i], beta*t_values[i]))
        Mat_fre, G = Edlib_to_G(G_data)
        wn = np.append(-np.flip(Mat_fre),Mat_fre)
        tau = np.linspace(0, beta, Ntau)
        neg_G = np.flip(G,axis=0)
        Gij_iwn = np.append(np.transpose(neg_G.conj(),axes=(0, 2, 1)),G[:],axis=0)
            
        Gij_iwn_data[i] = Gij_iwn
        Gij_tau_data[i] = Gij_iwn_to_Gij_tau(Gij_iwn, wn, tau, beta)
        Gk_tau_data[i] = np.real(Gij_to_Gk(Gij_tau_data[i]))
        Gk_iwn_data[i] = Gij_to_Gk(Gij_iwn)

        
    # Save data using pickle to handle lists of arrays with different shapes
    data_filename = f'{data_saving_directory}/ED_Ns{Ns}_beta{beta}.pkl'
    with open(data_filename, 'wb') as f:
        pickle.dump({'Gij_tau_data': Gij_tau_data, 'Gij_iwn_data': Gij_iwn_data, 'Gk_tau_data': Gk_tau_data, 'Gk_iwn_data': Gk_iwn_data}, f)
    return None

    
def load_data_training_G_ED(data_directory, Ns, beta):
    ### add to G0_data------------------
    data_filename = f'{data_directory}/ED_Ns{Ns}_beta{beta}.pkl'
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    Gij_tau_data = data['Gij_tau_data']
    Gij_iwn_data = data['Gij_iwn_data']
    Gk_tau_data = data['Gk_tau_data']
    Gk_iwn_data = data['Gk_iwn_data']
    
    return Gij_tau_data, Gij_iwn_data, Gk_tau_data, Gk_iwn_data