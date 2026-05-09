### Using greedy algorithm to find the optional AAA interpolation of G(t, iwn)

import warnings
import operator
import scipy.linalg
import math
import numpy as np
import scipy
from _bary_rational import AAA_adding0

def compare_l1(R):
    N = R.shape[0]
    
    # Pairwise L1 distances: shape (N, N)
    D = np.abs(R[:, None, :] - R[None, :, :]).sum(axis=2)
    
    # Ignore diagonal (distance to itself)
    np.fill_diagonal(D, np.inf)
    # Find argmin over all pairs
    i_min, j_min = np.unravel_index(np.argmin(D), D.shape)

    return i_min, j_min, D[i_min, j_min]

def G_interp(G_sce, t_sce, G_wce, t_wce, t_eva, N_part= 5, err = 1e-3, max_terms= 40):
    """ 
    G_wce : Nt_wce *1 

    """
    Nt_wce = t_wce.size
    Nt_sce = t_sce.size
    Nt_tot = t_eva.size 

    r_candidate = np.zeros((N_part+1, t_eva.size), dtype = complex)
    N_gap = t_sce.size // N_part
    
    ## Initial interpolation
    t_app = np.concatenate([t_sce[:1], t_wce], axis=0)
    G_app = np.concatenate([G_sce[:1], G_wce], axis=0)
    
    r = AAA_adding0(t_app, G_app,  rtol=err, max_terms= max_terms)
    r_candidate_supp = [r.support_points] 
    r_candidate[0,:] = r(t_eva)  

    
    k = 0 
    for n in range(0, t_sce.size, N_gap):
        print("n is", n)
        t_app = np.concatenate([t_sce[:n+1], t_wce], axis=0)
        G_app = np.concatenate([G_sce[:n+1], G_wce], axis=0)
        
        r = AAA_adding0(t_app, G_app,  rtol=err, max_terms= max_terms)

        ### Check whether the interpolation uses the same support points
        exists = False
        for v in r_candidate_supp:
            if np.array_equal(r.support_points, v):
                # print("The ",n, "-th candidate is the same, skip")
                exists = True
                break
                        
        ### Check whether the interpolation is real-pole free
        if not exists and np.min(np.abs(np.imag(r.poles()))) > 5*1e-2:
            # print("The ", n, "-th candidate is different and does not have real poles, keep")
            k = k + 1
            r_candidate_supp.append(r.support_points)
            r_candidate[k,:] = r(t_eva) 
           
    r_candidate = r_candidate[:k+1, :]   ## Cut the empty ones 
    if k == 0:
        best_interp = r_candidate[0, :]
        # print('Return baseline')
    else:
        i_min, j_min,  diff = compare_l1(r_candidate)
        best_interp = r_candidate[max(i_min,j_min), :]
        # print('Found Interpolation better than the baseline')
        # if diff<1e-2:
        #     best_interp = r_candidate[max(i_min,j_min), :]
        #     print('Found Interpolation better than the baseline')
        # else: 
        #     best_interp = r_candidate[0, :]
        #     print('Return baseline')
    return best_interp

def no_sign_change_check(v):
    v = np.asarray(v)
    no_change = not (np.any(v > 0) and np.any(v < 0))
    return no_change
    
# def G_interp_v2(G_sce, t_sce, G_wce, t_wce, t_eva, err = 1e-3, max_terms= 40):
#     """ 
#     G_wce : Nt_wce *1 

#     """
    
#     ## Initial interpolation
#     for n in range(t_sce.size):
#         t_app = np.concatenate([t_sce[:n+1], t_wce], axis=0)
#         G_app = np.concatenate([G_sce[:n+1], G_wce], axis=0)
#         r = AAA_adding0(t_app, G_app,  rtol=err, max_terms= max_terms)
#         tempo_interp =  r(t_eva) 
#         sign_change = has_sign_change(tempo_interp)
#         if np.min(np.abs(np.imag(r.poles()))) > 1e-2:
#             best_interp = tempo_interp
#         if not sign_change and np.min(np.abs(np.imag(r.poles()))) > 1e-2:
#             break
#     return best_interp

def G_interp(G_sce, t_sce, G_wce, t_wce, t_eva, err=1e-3, max_terms=40):
    """
    Retry logic:
      1) try with err
      2) if best_interp never assigned, retry with err = 1e-2
      3) all inputs are real
    """

    for current_err in (err, 5*err, 10*err):
        best_interp = None   # reset each attempt

        for n in range(t_sce.size):
            t_app = np.concatenate([t_sce[:n+1], t_wce], axis=0)
            G_app = np.concatenate([G_sce[:n+1], G_wce], axis=0)

            r = AAA_adding0(t_app, G_app, rtol=current_err,max_terms=max_terms)

            tempo_interp = r(t_eva)
            no_sign_change = no_sign_change_check(tempo_interp)

            # pole filter
            pole_ok = np.min(np.abs(np.imag(r.poles()))) > 1e-2

            if pole_ok:
                best_interp = tempo_interp

            if pole_ok and no_sign_change:
                # print("Best interpolation found")
                break

        # SUCCESS: stop retrying
        if best_interp is not None:
            return best_interp

    # If both attempts failed
    raise RuntimeError("G_interp_v2 failed: no valid interpolation found for err=1e-3 or err=1e-2")


def no_sign_change_check_complex(v):
    v = np.asarray(v)
    v1 = v.real
    v2 = v.imag

    real_no_change = not (np.any(v1 > 0) and np.any(v1 < 0))
    imag_no_change = not (np.any(v2 > 0) and np.any(v2 < 0))

    return real_no_change and imag_no_change

    
def Gk_interp(G_sce, t_sce, G_wce, t_wce, t_eva, err=1e-3, max_terms=40):
    """
    Retry logic:
      1) try with err
      2) if best_interp never assigned, retry with err = 1e-2
      3) the input can be complex
    """

    for current_err in (err, 5*err, 10*err):
        best_interp = None   # reset each attempt

        for n in range(t_sce.size):
            t_app = np.concatenate([t_sce[:n+1], t_wce], axis=0)
            G_app = np.concatenate([G_sce[:n+1], G_wce], axis=0)

            r = AAA_adding0(t_app, G_app, rtol=current_err,max_terms=max_terms)

            tempo_interp = r(t_eva)
            no_sign_change = no_sign_change_check_complex(tempo_interp)

            # pole filter
            pole_ok = np.min(np.abs(np.imag(r.poles()))) > 1e-2

            if pole_ok:
                best_interp = tempo_interp

            if pole_ok and no_sign_change:
                # print("Best interpolation found")
                break

        # SUCCESS: stop retrying
        if best_interp is not None:
            return best_interp

    # If both attempts failed
    raise RuntimeError("G_interp_v2 failed: no valid interpolation found for err=1e-3 or err=1e-2")