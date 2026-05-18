"""Helpers for Hubbard-dimer interpolation data on complex t grids."""

import numpy as np

try:
    import analytic_solution as AS
    import diff_G as diff
except ImportError:  # pragma: no cover - used when imported as a package.
    from . import analytic_solution as AS
    from . import diff_G as diff


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
):
    """Build approximation data on an arbitrary complex ``t`` grid.

    Parameters
    ----------
    sce_order, wce_order : int
        Expansion orders used for the strong- and weak-coupling approximations.
    baseline_G : array_like, shape (Nt, 2, 2)
        Exact ``G_ij`` for one Matsubara frequency on ``t_values``.
    t_values : array_like, shape (Nt,)
        Complex grid points, for example lattice points inside a tube.
    epsilon : float
        Error bar for deciding whether SCE or WCE is valid at each point.
    wn : float
        The single Matsubara frequency for this one-frequency slice.
    U, beta : float
        Hubbard interaction and inverse temperature.

    Returns
    -------
    t_app : ndarray, shape (Napp,)
        Complex t points where SCE or WCE approximates ``baseline_G`` within
        ``epsilon``.
    G_app : ndarray, shape (Napp, 2, 2)
        The corresponding SCE/WCE values. If both are valid at a point, the one
        with the smaller matrix max error is used.
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

    sce_ok = sce_error <= epsilon
    wce_ok = wce_error <= epsilon
    app_ok = sce_ok | wce_ok

    t_app = t_values[app_ok]
    G_app = np.empty((t_app.size, 2, 2), dtype=complex)

    use_sce = sce_ok & (~wce_ok | (sce_error <= wce_error))
    use_wce = wce_ok & ~use_sce

    G_choice = np.empty_like(baseline_G, dtype=complex)
    G_choice[use_sce] = G_sce[use_sce]
    G_choice[use_wce] = G_wce[use_wce]
    G_app[:] = G_choice[app_ok]

    print(
        "Selected complex t points:",
        int(app_ok.sum()),
        "of",
        int(t_values.size),
        "| SCE valid:",
        int(sce_ok.sum()),
        "| WCE valid:",
        int(wce_ok.sum()),
    )

    return t_app, G_app


def _matrix_error(approx_G, baseline_G):
    return np.nan_to_num(
        np.max(np.abs(approx_G - baseline_G), axis=(1, 2)),
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
