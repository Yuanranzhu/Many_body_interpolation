"""Small helpers for Hubbard-dimer t interpolation data."""

import numpy as np

try:
    import analytic_solution as AS
    import diff_G as diff
    from _bary_rational import AAA
except ImportError:  # pragma: no cover - used when imported as a package.
    from . import analytic_solution as AS
    from . import diff_G as diff
    from ._bary_rational import AAA


def compute_exact_t(t_values, wn, U, beta):
    """Compute exact ``G_ij(t, iwn)`` on ``t_values``.

    Returns an array with shape ``(len(t_values), len(wn), 2, 2)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    G = np.zeros((t_values.size, wn.size, 2, 2), dtype=complex)

    for i, t in enumerate(t_values):
        G[i] = AS.G_analytic(t, wn, U, beta)
    return G


def compute_wce_t(order, t_values, wn, U):
    """Compute WCE ``G_ij(t, iwn)`` at a given order on ``t_values``.

    Returns an array with shape ``(len(t_values), len(wn), 2, 2)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    G = np.zeros((t_values.size, wn.size, 2, 2), dtype=complex)

    for i, t in enumerate(t_values):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            G[i] = diff.Gij_wce_series(order, t, U, wn)
    return G


def compute_sce_t(order, t_values, wn, U, beta):
    """Compute SCE ``G_ij(t, iwn)`` at a given order on ``t_values``.

    Returns an array with shape ``(len(t_values), len(wn), 2, 2)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    G = np.zeros((t_values.size, wn.size, 2, 2), dtype=complex)

    for i, t in enumerate(t_values):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            G[i] = diff.Gij_sce_series(order, t, U, wn, beta)
    return G


def build_t_app_G_app(
    sce_order,
    wce_order,
    baseline_G,
    t_values,
    epsilon,
    *,
    wn,
    U,
    beta,
    fine_grid_size=1001,
    aaa_resample=True,
):
    """Build interpolation input data for one Matsubara frequency.

    Parameters
    ----------
    sce_order, wce_order : int
        Expansion orders used for the strong- and weak-coupling approximations.
    baseline_G : array_like, shape (Nt, 2, 2)
        Exact ``G_ij`` for one Matsubara frequency on ``t_values``.
    t_values : array_like, shape (Nt,)
        Grid used by ``baseline_G``. This can be real or a horizontal complex
        line ``real(t) + 1j * shift``.
    epsilon : float
        Error bar for deciding valid SCE/WCE windows.
    wn : float
        The single Matsubara frequency for this one-frequency slice.
    U, beta : float
        Hubbard interaction and inverse temperature.
    fine_grid_size : int, optional
        Number of fine-grid points used for each valid domain after separately
        AAA-interpolating the SCE and WCE data. When both domains are present,
        ``len(t_app) == 2 * fine_grid_size``.
    aaa_resample : bool, optional
        If True, AAA-resample the valid SCE and WCE domains onto fine grids. If
        False, return the valid SCE/WCE data directly on the original ``t_values``
        grid.

    Returns
    -------
    t_app : ndarray, shape (Napp,)
        Concatenated t points from the SCE-valid prefix and WCE-valid suffix.
    G_app : ndarray, shape (Napp, 2, 2)
        Concatenated SCE/WCE Green's-function values on ``t_app``.
    missing_t_domain : tuple[float, float]
        The t interval between the SCE-valid domain and the WCE-valid domain. If
        the domains overlap, this is returned as a degenerate interval.
    """
    t_values = np.asarray(t_values)
    baseline_G = np.asarray(baseline_G)
    wn_array = np.asarray([wn])

    if baseline_G.shape != (t_values.size, 2, 2):
        raise ValueError("`baseline_G` must have shape (len(t_values), 2, 2).")

    G_sce = compute_sce_t(sce_order, t_values, wn_array, U, beta)[:, 0]
    G_wce = compute_wce_t(wce_order, t_values, wn_array, U)[:, 0]

    sce_error = _matrix_error(G_sce, baseline_G)
    wce_error = _matrix_error(G_wce, baseline_G)

    t_real = np.real(t_values)
    if np.any(np.diff(t_real) < 0):
        raise ValueError("`real(t_values)` must be sorted in nondecreasing order.")

    sce_idx = _last_contiguous_true_from_start(sce_error <= epsilon)
    wce_idx = _first_contiguous_true_from_end(wce_error <= epsilon)

    if sce_idx is None:
        raise ValueError("No contiguous SCE-valid t window found.")
    if wce_idx is None:
        raise ValueError("No contiguous WCE-valid t window found.")

    print(
        "SCE t domain:",
        (float(np.real(t_values[0])), float(np.real(t_values[sce_idx]))),
        "WCE t domain:",
        (float(np.real(t_values[wce_idx])), float(np.real(t_values[-1]))),
    )

    if sce_idx < wce_idx:
        missing_t_domain = (
            float(np.real(t_values[sce_idx])),
            float(np.real(t_values[wce_idx])),
        )
    else:
        missing_t_domain = (
            float(np.real(t_values[wce_idx])),
            float(np.real(t_values[wce_idx])),
        )
    print("Missing t domain:", missing_t_domain)

    if aaa_resample:
        fine_grid_size = _validate_fine_grid_size(fine_grid_size)

        t_sce_app, G_sce_app = _aaa_resample_matrix(
            t_values[: sce_idx + 1],
            G_sce[: sce_idx + 1],
            fine_grid_size,
        )
        t_wce_app, G_wce_app = _aaa_resample_matrix(
            t_values[wce_idx:],
            G_wce[wce_idx:],
            fine_grid_size,
        )
    else:
        t_sce_app = t_values[: sce_idx + 1]
        G_sce_app = G_sce[: sce_idx + 1]
        t_wce_app = t_values[wce_idx:]
        G_wce_app = G_wce[wce_idx:]

    t_app = np.concatenate([t_sce_app, t_wce_app], axis=0)
    G_app = np.concatenate([G_sce_app, G_wce_app], axis=0)
    return t_app, G_app, missing_t_domain


def _validate_fine_grid_size(fine_grid_size):
    fine_grid_size = int(fine_grid_size)
    if fine_grid_size < 1:
        raise ValueError("`fine_grid_size` must be at least one.")
    return fine_grid_size


def _aaa_resample_matrix(t_domain, G_domain, n_fine):
    if n_fine == 0:
        return np.empty(0, dtype=complex), np.empty((0, 2, 2), dtype=complex)

    t_domain = np.asarray(t_domain)
    G_domain = np.asarray(G_domain)
    if t_domain.size == 0:
        raise ValueError("Cannot resample an empty t domain.")

    if t_domain.size == 1:
        t_fine = np.full(n_fine, t_domain[0], dtype=complex)
        G_fine = np.repeat(G_domain[:1], n_fine, axis=0)
        return t_fine, G_fine

    t_real_fine = np.linspace(np.real(t_domain[0]), np.real(t_domain[-1]), n_fine)
    t_imag_fine = np.linspace(np.imag(t_domain[0]), np.imag(t_domain[-1]), n_fine)
    t_fine = t_real_fine + 1j * t_imag_fine
    G_fine = np.zeros((n_fine, 2, 2), dtype=complex)
    max_terms = min(100, t_domain.size)

    for i in range(2):
        for j in range(2):
            r = AAA(
                t_domain,
                G_domain[:, i, j],
                rtol=1e-13,
                max_terms=max_terms,
                clean_up=False,
            )
            G_fine[:, i, j] = r(t_fine)

    return t_fine, G_fine


def _matrix_error(approx_G, baseline_G):
    return np.nan_to_num(
        np.max(np.abs(approx_G - baseline_G), axis=(1, 2)),
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )


def _last_contiguous_true_from_start(ok):
    if ok.size == 0 or not ok[0]:
        return None
    false_idx = np.flatnonzero(~ok)
    return ok.size - 1 if false_idx.size == 0 else false_idx[0] - 1


def _first_contiguous_true_from_end(ok):
    if ok.size == 0 or not ok[-1]:
        return None
    false_from_end = np.flatnonzero(~ok[::-1])
    return 0 if false_from_end.size == 0 else ok.size - false_from_end[0]
