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

    Returns an array with shape ``(len(t_values), len(wn), Ns, Ns)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    if t_values.size == 0:
        return np.zeros((0, wn.size, 0, 0), dtype=complex)

    first = np.asarray(AS.G_analytic(t_values[0], wn, U, beta), dtype=complex)
    _validate_frequency_matrix_shape(first, wn, "AS.G_analytic")

    G = np.empty((t_values.size,) + first.shape, dtype=complex)
    G[0] = first

    for i, t in enumerate(t_values[1:], start=1):
        G[i] = AS.G_analytic(t, wn, U, beta)
    return G


def compute_wce_t(order, t_values, wn, U):
    """Compute WCE ``G_ij(t, iwn)`` at a given order on ``t_values``.

    Returns an array with shape ``(len(t_values), len(wn), Ns, Ns)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    if t_values.size == 0:
        return np.zeros((0, wn.size, 0, 0), dtype=complex)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        first = np.asarray(diff.Gij_wce_series(order, t_values[0], U, wn), dtype=complex)
    _validate_frequency_matrix_shape(first, wn, "diff.Gij_wce_series")

    G = np.empty((t_values.size,) + first.shape, dtype=complex)
    G[0] = first

    for i, t in enumerate(t_values[1:], start=1):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            G[i] = diff.Gij_wce_series(order, t, U, wn)
    return G


def compute_sce_t(order, t_values, wn, U, beta):
    """Compute SCE ``G_ij(t, iwn)`` at a given order on ``t_values``.

    Returns an array with shape ``(len(t_values), len(wn), Ns, Ns)``.
    """
    t_values = np.asarray(t_values)
    wn = np.asarray(wn)
    if t_values.size == 0:
        return np.zeros((0, wn.size, 0, 0), dtype=complex)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        first = np.asarray(diff.Gij_sce_series(order, t_values[0], U, wn, beta), dtype=complex)
    _validate_frequency_matrix_shape(first, wn, "diff.Gij_sce_series")

    G = np.empty((t_values.size,) + first.shape, dtype=complex)
    G[0] = first

    for i, t in enumerate(t_values[1:], start=1):
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
    baseline_G : array_like, shape (Nt, Ns, Ns)
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
    G_app : ndarray, shape (Napp, Ns, Ns)
        The corresponding SCE/WCE values. If both are valid at a point, the one
        with the smaller matrix max error is used.
    """
    t_values = np.asarray(t_values)
    baseline_G = np.asarray(baseline_G)
    wn_array = np.asarray([wn])

    if baseline_G.ndim != 3:
        raise ValueError("`baseline_G` must have shape (len(t_values), Ns, Ns).")
    if baseline_G.shape[0] != t_values.size:
        raise ValueError("`baseline_G` must have first dimension len(t_values).")
    if baseline_G.shape[1] != baseline_G.shape[2]:
        raise ValueError("`baseline_G` must have square matrix slices.")

    G_sce = compute_sce_t(sce_order, t_values, wn_array, U, beta)[:, 0]
    G_wce = compute_wce_t(wce_order, t_values, wn_array, U)[:, 0]

    if G_sce.shape != baseline_G.shape:
        raise ValueError("Computed SCE data shape does not match `baseline_G`.")
    if G_wce.shape != baseline_G.shape:
        raise ValueError("Computed WCE data shape does not match `baseline_G`.")

    return build_t_app_G_app_from_data(G_sce, G_wce, baseline_G, t_values, epsilon)


def build_t_app_G_app_scalar(
    G_sce,
    G_wce,
    baseline_G,
    t_values,
    epsilon,
):
    """Build scalar approximation data on an arbitrary complex ``t`` grid.

    Parameters
    ----------
    G_sce, G_wce, baseline_G : array_like, shape (Nt,)
        Scalar data for one matrix element at one Matsubara frequency on
        ``t_values``.
    t_values : array_like, shape (Nt,)
        Complex grid points, for example lattice points inside a tube.
    epsilon : float
        Error bar for deciding whether SCE or WCE is valid at each point.

    Returns
    -------
    t_app : ndarray, shape (Napp,)
        Complex t points where SCE or WCE approximates ``baseline_G`` within
        ``epsilon``.
    G_app : ndarray, shape (Napp,)
        The corresponding SCE/WCE scalar values. If both are valid at a point,
        the one with the smaller scalar error is used.
    """
    t_values = np.asarray(t_values)
    G_sce = np.asarray(G_sce)
    G_wce = np.asarray(G_wce)
    baseline_G = np.asarray(baseline_G)

    expected_shape = (t_values.size,)
    if G_sce.shape != expected_shape:
        raise ValueError("`G_sce` must have shape (len(t_values),).")
    if G_wce.shape != expected_shape:
        raise ValueError("`G_wce` must have shape (len(t_values),).")
    if baseline_G.shape != expected_shape:
        raise ValueError("`baseline_G` must have shape (len(t_values),).")

    sce_error = _scalar_error(G_sce, baseline_G)
    wce_error = _scalar_error(G_wce, baseline_G)

    sce_ok = sce_error <= epsilon
    wce_ok = wce_error <= epsilon
    app_ok = sce_ok | wce_ok

    use_sce = sce_ok & (~wce_ok | (sce_error <= wce_error))
    use_wce = wce_ok & ~use_sce

    G_choice = np.empty_like(baseline_G, dtype=complex)
    G_choice[use_sce] = G_sce[use_sce]
    G_choice[use_wce] = G_wce[use_wce]

    t_app = t_values[app_ok]
    G_app = G_choice[app_ok]

    print(
        "Selected scalar complex t points:",
        int(app_ok.sum()),
        "of",
        int(t_values.size),
        "| SCE valid:",
        int(sce_ok.sum()),
        "| WCE valid:",
        int(wce_ok.sum()),
    )

    return t_app, G_app


def build_t_app_G_app_from_data(
    G_sce,
    G_wce,
    baseline_G,
    t_values,
    epsilon,
):
    """Build matrix approximation data from precomputed SCE/WCE arrays.

    This is the data-driven version of ``build_t_app_G_app``: it does not
    compute SCE or WCE internally.

    Parameters
    ----------
    G_sce, G_wce, baseline_G : array_like, shape (Nt, Ns, Ns)
        Matrix-valued data for one Matsubara frequency on ``t_values``.
    t_values : array_like, shape (Nt,)
        Complex grid points, for example lattice points inside a tube.
    epsilon : float
        Error bar for deciding whether SCE or WCE is valid at each point.

    Returns
    -------
    t_app : ndarray, shape (Napp,)
        Complex t points where SCE or WCE approximates ``baseline_G`` within
        ``epsilon``.
    G_app : ndarray, shape (Napp, Ns, Ns)
        The corresponding SCE/WCE matrix values. If both are valid at a point,
        the one with the smaller matrix max error is used.
    """
    t_values = np.asarray(t_values)
    G_sce = np.asarray(G_sce)
    G_wce = np.asarray(G_wce)
    baseline_G = np.asarray(baseline_G)

    if baseline_G.ndim != 3:
        raise ValueError("`baseline_G` must have shape (len(t_values), Ns, Ns).")
    if baseline_G.shape[0] != t_values.size:
        raise ValueError("`baseline_G` must have first dimension len(t_values).")
    if baseline_G.shape[1] != baseline_G.shape[2]:
        raise ValueError("`baseline_G` must have square matrix slices.")
    if G_sce.shape != baseline_G.shape:
        raise ValueError("`G_sce` must have the same shape as `baseline_G`.")
    if G_wce.shape != baseline_G.shape:
        raise ValueError("`G_wce` must have the same shape as `baseline_G`.")

    sce_error = _matrix_error(G_sce, baseline_G)
    wce_error = _matrix_error(G_wce, baseline_G)

    sce_ok = sce_error <= epsilon
    wce_ok = wce_error <= epsilon
    app_ok = sce_ok | wce_ok

    use_sce = sce_ok & (~wce_ok | (sce_error <= wce_error))
    use_wce = wce_ok & ~use_sce

    G_choice = np.empty_like(baseline_G, dtype=complex)
    G_choice[use_sce] = G_sce[use_sce]
    G_choice[use_wce] = G_wce[use_wce]

    t_app = t_values[app_ok]
    G_app = G_choice[app_ok]

    print(
        "Selected matrix complex t points:",
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


def _validate_frequency_matrix_shape(G, wn, source_name):
    if G.ndim != 3:
        raise ValueError(
            f"`{source_name}` must return an array with shape (len(wn), Ns, Ns)."
        )
    if G.shape[0] != wn.size:
        raise ValueError(
            f"`{source_name}` first dimension must match len(wn)."
        )
    if G.shape[1] != G.shape[2]:
        raise ValueError(f"`{source_name}` must return square matrix slices.")


def _scalar_error(approx_G, baseline_G):
    return np.nan_to_num(
        np.abs(approx_G - baseline_G),
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
