#### Calculating the 2ndB self-energy for half-filled Hubbard model by direct tensor contraction in Matsubara freq 
import numpy as np

def fermionic_n_vals(Nw_half):
    """
    Integer labels n for fermionic Matsubara frequencies ω_n = (2n+1)π/β
    using n = -Nw_half, ..., Nw_half-1  (total Nw = 2*Nw_half).
    """
    return np.arange(-Nw_half, Nw_half, dtype=int)

def _2ndB_from_G0(G0, U, beta, n_vals, overall_sign=1.0):
    r"""
    Second-order Born self-energy in real space, Matsubara summation:

      Σ^(2)_{ij}(n) = overall_sign * (U^2/β^2) Σ_{n1,n2}
                        G0_{ij}(n1) G0_{ij}(n2) G0_{ji}(n1+n2-n)

    Parameters
    ----------
    G0 : complex ndarray, shape (Nw, Ns, Ns)
        Lattice bare Green's function G^0_{ij}(iω_n) on a fermionic grid.
        Must include both positive and negative fermionic frequencies.
    U : float
        Hubbard interaction.
    beta : float
        Inverse temperature.
    n_vals : int ndarray, shape (Nw,)
        Integer Matsubara labels corresponding to the first axis of G0, i.e.
        ω_n = (2*n_vals + 1)*π/beta.
        Must be consecutive and unique (e.g. [-N..N-1]).
    overall_sign : float
        Use +1 or -1 depending on your sign convention.

    Returns
    -------
    Sigma2 : complex ndarray, shape (Nw, Ns, Ns)
        Second-order self-energy Σ^(2)_{ij}(iω_n).
    """
    G0 = np.asarray(G0)
    if G0.ndim != 3:
        raise ValueError("G0 must have shape (Nw, Ns, Ns).")
    Nw, Ns1, Ns2 = G0.shape
    if Ns1 != Ns2:
        raise ValueError("G0 must be square in site indices (Ns, Ns).")
    n_vals = np.asarray(n_vals, dtype=int)
    if n_vals.shape != (Nw,):
        raise ValueError("n_vals must have shape (Nw,) matching G0's first axis.")

    # Map Matsubara integer label -> frequency-axis index
    n_to_idx = {int(n): i for i, n in enumerate(n_vals)}

    # Precompute grids for n1, n2 (broadcastable)
    n1 = n_vals[:, None]          # (Nw,1)
    n2 = n_vals[None, :]          # (1,Nw)

    Sigma2 = np.zeros((Nw, Ns1, Ns1), dtype=np.complex128)
    pref = overall_sign * (U * U) / (beta * beta)

    # Compute Σ(n_out) for each output fermionic index
    for out_idx, n_out in enumerate(n_vals):
        # n3 = n1 + n2 - n_out  (shape Nw x Nw)
        n3 = n1 + n2 - int(n_out)

        # Build index array for n3; mask out-of-range terms safely
        idx3 = np.full((Nw, Nw), -1, dtype=int)
        valid = np.isin(n3, n_vals)
        if np.any(valid):
            # vectorized fill using mapping via sorting trick
            # (fast for consecutive n_vals too)
            # For consecutive n_vals, you can do idx = n3 - n_vals[0] instead.
            n_min = int(n_vals.min())
            n_max = int(n_vals.max())
            consecutive = np.array_equal(n_vals, np.arange(n_min, n_max + 1, dtype=int))
            if consecutive:
                idx3[valid] = (n3[valid] - n_min).astype(int)
            else:
                # generic mapping
                flat = n3[valid].ravel()
                idx3[valid] = np.array([n_to_idx[int(x)] for x in flat], dtype=int).reshape(-1)

        # Gather third Green's function factor: G0_{ji}(n3)
        # Shape: (Nw, Nw, Ns, Ns)
        G3 = np.zeros((Nw, Nw, Ns1, Ns1), dtype=np.complex128)
        if np.any(valid):
            G3_valid = G0[idx3[valid], :, :].reshape(-1, Ns1, Ns1)  # (Nv, Ns, Ns) for n3
            # But we need G0_{ji}, so transpose site indices:
            G3_valid = np.swapaxes(G3_valid, -1, -2)                # (Nv, Ns, Ns) now ji
            G3[valid] = G3_valid

        # Two factors G0_{ij}(n1) and G0_{ij}(n2), broadcast to (Nw,Nw,Ns,Ns)
        A = G0[:, None, :, :]   # (Nw,1,Ns,Ns) -> n1
        B = G0[None, :, :, :]   # (1,Nw,Ns,Ns) -> n2

        # Sum over n1,n2 with tensor contraction:
        # Σ_{ij} = pref * Σ_{n1,n2} A(n1,ij)*B(n2,ij)*G3(n1,n2,ij)
        # einsum indices: a=n1, b=n2, i,j=sites
        Sigma2[out_idx] = pref * np.einsum("abij,abij->ij", A * B, G3, optimize=True)

    return Sigma2


def _2ndB_from_G0_lean(G0, U, beta, n_vals, overall_sign = 1.0, assume_consecutive = True):
    r"""
    Memory-lean direct Matsubara evaluation of

      Σ^(2)_{ij}(n) = overall_sign * (U^2/β^2) Σ_{n1,n2}
                        G0_{ij}(n1) G0_{ij}(n2) G0_{ji}(n1+n2-n)

    Uses: for each output n and each n1, do the n2-sum by einsum over (n2,i,j).
    Peak memory: O(Nw*Ns^2) rather than O(Nw^2*Ns^2).
    """
    G0 = np.asarray(G0)
    if G0.ndim != 3:
        raise ValueError("G0 must have shape (Nw, Ns, Ns).")
    Nw, Ns1, Ns2 = G0.shape
    if Ns1 != Ns2:
        raise ValueError("G0 must be square in site indices (Ns, Ns).")

    n_vals = np.asarray(n_vals, dtype=int)
    if n_vals.shape != (Nw,):
        raise ValueError("n_vals must have shape (Nw,) matching G0's first axis.")

    n_min = int(n_vals.min())
    n_max = int(n_vals.max())
    if assume_consecutive:
        if not np.array_equal(n_vals, np.arange(n_min, n_max + 1, dtype=int)):
            raise ValueError("n_vals is not consecutive; set assume_consecutive=False.")
        # fast mapping for consecutive n_vals
        def idx_of(n_int: np.ndarray) -> np.ndarray:
            return (n_int - n_min).astype(int)
        def in_range(n_int: np.ndarray) -> np.ndarray:
            return (n_int >= n_min) & (n_int <= n_max)
    else:
        n_to_idx = {int(n): i for i, n in enumerate(n_vals)}
        def idx_of(n_int: np.ndarray) -> np.ndarray:
            flat = n_int.ravel()
            mapped = np.array([n_to_idx.get(int(x), -1) for x in flat], dtype=int)
            return mapped.reshape(n_int.shape)
        def in_range(n_int: np.ndarray) -> np.ndarray:
            # generic membership
            return np.isin(n_int, n_vals)

    pref = overall_sign * (U * U) / (beta * beta)
    Sigma2 = np.zeros((Nw, Ns1, Ns1), dtype=np.complex128)

    # Precompute transpose in site indices once: G0T[n] = G0[n]^T = G0_{ji}(n)
    G0T = np.swapaxes(G0, -1, -2)

    n2_vals = n_vals  # alias

    for out_idx, n_out in enumerate(n_vals):
        acc = np.zeros((Ns1, Ns1), dtype=np.complex128)

        # Loop over n1; for each n1 build the needed n3 array over n2
        for i1, n1 in enumerate(n_vals):
            n3 = (n1 + n2_vals - int(n_out))          # shape (Nw,)
            valid = in_range(n3)                       # shape (Nw,)
            if not np.any(valid):
                continue

            idx3 = idx_of(n3[valid])                  # shape (Nv,)

            # For this n1: A_ij = G0_ij(n1) is (Ns,Ns)
            A = G0[i1]                                 # (Ns,Ns)

            # For valid n2: B(n2,ij)=G0(n2,ij) and C(n2,ij)=G0T(n3,ij)
            # Shapes: (Nv,Ns,Ns)
            B = G0[valid]                              # (Nv,Ns,Ns)
            C = G0T[idx3]                              # (Nv,Ns,Ns)  = G0_{ji}(n3)

            # Sum over n2: sum_{n2} B(n2,ij)*C(n2,ij) -> (Ns,Ns)
            BCsum = np.einsum("vij,vij->ij", B, C, optimize=True)

            # Then multiply by A_ij and accumulate over n1
            acc += A * BCsum

        Sigma2[out_idx] = pref * acc

    return Sigma2


def _2ndB_from_G0_blocked(G0, U, beta, n_vals, overall_sign = 1.0, block_n2 = 32):
    """
    Same Σ^(2) as above, but blocks the n2 contraction to cap memory.

    block_n2: number of n2 frequencies per block.
    """
    G0 = np.asarray(G0)
    Nw, Ns, Ns2 = G0.shape
    if Ns != Ns2:
        raise ValueError("G0 must have shape (Nw, Ns, Ns).")
    n_vals = np.asarray(n_vals, dtype=int)
    if n_vals.shape != (Nw,):
        raise ValueError("n_vals must match G0 frequency axis.")
    n_min = int(n_vals.min())
    n_max = int(n_vals.max())
    if not np.array_equal(n_vals, np.arange(n_min, n_max + 1, dtype=int)):
        raise ValueError("This blocked version assumes consecutive n_vals.")

    pref = overall_sign * (U * U) / (beta * beta)
    Sigma2 = np.zeros((Nw, Ns, Ns), dtype=np.complex128)
    G0T = np.swapaxes(G0, -1, -2)

    for out_idx, n_out in enumerate(n_vals):
        acc = np.zeros((Ns, Ns), dtype=np.complex128)
        for i1, n1 in enumerate(n_vals):
            A = G0[i1]  # (Ns,Ns)

            # n3(n2) = n1+n2-n_out
            n3_all = n1 + n_vals - int(n_out)  # (Nw,)

            # Only n3 in range contribute (here range is [n_min,n_max])
            valid_all = (n3_all >= n_min) & (n3_all <= n_max)
            if not np.any(valid_all):
                continue

            # We'll traverse n2 indices in blocks, but skip invalid ones
            # to avoid assembling large B/C.
            valid_idx = np.nonzero(valid_all)[0]  # positions in n2 axis

            # block over the *valid* n2 positions
            for s in range(0, valid_idx.size, block_n2):
                blk = valid_idx[s:s+block_n2]
                idx3 = (n3_all[blk] - n_min).astype(int)

                B = G0[blk]        # (b,Ns,Ns)
                C = G0T[idx3]      # (b,Ns,Ns)

                acc += A * np.einsum("bij,bij->ij", B, C, optimize=True)

        Sigma2[out_idx] = pref * acc

    return Sigma2

def G2(G0, U, beta, n_vals, overall_sign=1.0):
    # Sigma = _2ndB_from_G0(G0, U, beta, n_vals, overall_sign=1.0)
    Sigma = _2ndB_from_G0_lean(G0, U, beta, n_vals, overall_sign=1.0)
    G = G0 + G0@Sigma@G0
    return G











    
