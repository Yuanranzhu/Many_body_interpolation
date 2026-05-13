# Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import operator
import scipy.linalg
import math
import numpy as np
import scipy


__all__ = [
    "AAA",
    "AAA_filter",
    "AAA_adding0",
    "AAA_holo_interp",
    "AAA_holo_interp_v2",
    "AAA_tube",
    "FloaterHormannInterpolator",
]


class _BarycentricRational:
    """Base class for Barycentric representation of a rational function."""
    def __init__(self, x, y, **kwargs):
        # input validation
        z = np.asarray(x)
        f = np.asarray(y)

        self._input_validation(z, f, **kwargs)

        # Remove infinite or NaN function values and repeated entries
        to_keep = np.logical_and.reduce(
            ((np.isfinite(f)) & (~np.isnan(f))).reshape(f.shape[0], -1),
            axis=-1
        )
        f = f[to_keep, ...]
        z = z[to_keep]
        z, uni = np.unique(z, return_index=True)
        f = f[uni, ...]

        self._shape = f.shape[1:]
        self._support_points, self._support_values, self.weights = (
            self._compute_weights(z, f, **kwargs)
        )

        # only compute once
        self._poles = None
        self._residues = None
        self._roots = None

    def _input_validation(self, x, y, **kwargs):
        if x.ndim != 1:
            raise ValueError("`x` must be 1-D.")

        if not y.ndim >= 1:
            raise ValueError("`y` must be at least 1-D.")

        if x.size != y.shape[0]:
            raise ValueError("`x` be the same size as the first dimension of `y`.")

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must be finite.")

    def _compute_weights(z, f, **kwargs):
        raise NotImplementedError

    def __call__(self, z):
        """Evaluate the rational approximation at given values.

        Parameters
        ----------
        z : array_like
            Input values.
        """
        # evaluate rational function in barycentric form.
        z = np.asarray(z)
        zv = np.ravel(z)

        support_values = self._support_values.reshape(
            (self._support_values.shape[0], -1)
        )
        weights = self.weights[..., np.newaxis]

        # Cauchy matrix
        # Ignore errors due to inf/inf at support points, these will be fixed later
        with np.errstate(invalid="ignore", divide="ignore"):
            CC = 1 / np.subtract.outer(zv, self._support_points)
            # Vector of values
            r = CC @ (weights * support_values) / (CC @ weights)

        # Deal with input inf: `r(inf) = lim r(z) = sum(w*f) / sum(w)`
        if np.any(np.isinf(zv)):
            r[np.isinf(zv)] = (np.sum(weights * support_values)
                               / np.sum(weights))

        # Deal with NaN
        ii = np.nonzero(np.isnan(r))[0]
        for jj in ii:
            if np.isnan(zv[jj]) or not np.any(zv[jj] == self._support_points):
                # r(NaN) = NaN is fine.
                # The second case may happen if `r(zv[ii]) = 0/0` at some point.
                pass
            else:
                # Clean up values `NaN = inf/inf` at support points.
                # Find the corresponding node and set entry to correct value:
                r[jj] = support_values[zv[jj] == self._support_points].squeeze()

        return np.reshape(r, z.shape + self._shape)

    def poles(self):
        """Compute the poles of the rational approximation.

        Returns
        -------
        poles : array
            Poles of the AAA approximation, repeated according to their multiplicity
            but not in any specific order.
        """
        if self._poles is None:
            # Compute poles via generalized eigenvalue problem
            m = self.weights.size
            B = np.eye(m + 1, dtype=self.weights.dtype)
            B[0, 0] = 0

            E = np.zeros_like(B, dtype=np.result_type(self.weights,
                                                      self._support_points))
            E[0, 1:] = self.weights
            E[1:, 0] = 1
            np.fill_diagonal(E[1:, 1:], self._support_points)

            pol = scipy.linalg.eigvals(E, B)
            self._poles = pol[np.isfinite(pol)]
        return self._poles

    def residues(self):
        """Compute the residues of the poles of the approximation.

        Returns
        -------
        residues : array
            Residues associated with the `poles` of the approximation
        """
        if self._residues is None:
            # Compute residues via formula for res of quotient of analytic functions
            with np.errstate(divide="ignore", invalid="ignore"):
                N = (1/(np.subtract.outer(self.poles(), self._support_points))) @ (
                    self._support_values * self.weights
                )
                Ddiff = (
                    -((1/np.subtract.outer(self.poles(), self._support_points))**2)
                    @ self.weights
                )
                self._residues = N / Ddiff
        return self._residues

    def roots(self):
        """Compute the zeros of the rational approximation.

        Returns
        -------
        zeros : array
            Zeros of the AAA approximation, repeated according to their multiplicity
            but not in any specific order.
        """
        if self._roots is None:
            # Compute zeros via generalized eigenvalue problem
            m = self.weights.size
            B = np.eye(m + 1, dtype=self.weights.dtype)
            B[0, 0] = 0
            E = np.zeros_like(B, dtype=np.result_type(self.weights,
                                                      self._support_values,
                                                      self._support_points))
            E[0, 1:] = self.weights * self._support_values
            E[1:, 0] = 1
            np.fill_diagonal(E[1:, 1:], self._support_points)

            zer = scipy.linalg.eigvals(E, B)
            self._roots = zer[np.isfinite(zer)]
        return self._roots

class AAA(_BarycentricRational):
    r"""
    AAA real or complex rational approximation.

    As described in [1]_, the AAA algorithm is a greedy algorithm for approximation by
    rational functions on a real or complex set of points. The rational approximation is
    represented in a barycentric form from which the roots (zeros), poles, and residues
    can be computed.

    Parameters
    ----------
    x : 1D array_like, shape (n,)
        1-D array containing values of the independent variable. Values may be real or
        complex but must be finite.
    y : 1D array_like, shape (n,)
        Function values ``f(x)``. Infinite and NaN values of `values` and
        corresponding values of `points` will be discarded.
    rtol : float, optional
        Relative tolerance, defaults to ``eps**0.75``. If a small subset of the entries
        in `values` are much larger than the rest the default tolerance may be too
        loose. If the tolerance is too tight then the approximation may contain
        Froissart doublets or the algorithm may fail to converge entirely.
    max_terms : int, optional
        Maximum number of terms in the barycentric representation, defaults to ``100``.
        Must be greater than or equal to one.
    clean_up : bool, optional
        Automatic removal of Froissart doublets, defaults to ``True``. See notes for
        more details.
    clean_up_tol : float, optional
        Poles with residues less than this number times the geometric mean
        of `values` times the minimum distance to `points` are deemed spurious by the
        cleanup procedure, defaults to 1e-13. See notes for more details.

    Attributes
    ----------
    support_points : array
        Support points of the approximation. These are a subset of the provided `x` at
        which the approximation strictly interpolates `y`.
        See notes for more details.
    support_values : array
        Value of the approximation at the `support_points`.
    weights : array
        Weights of the barycentric approximation.
    errors : array
        Error :math:`|f(z) - r(z)|_\infty` over `points` in the successive iterations
        of AAA.

    Warns
    -----
    RuntimeWarning
        If `rtol` is not achieved in `max_terms` iterations.

    See Also
    --------
    FloaterHormannInterpolator : Floater-Hormann barycentric rational interpolation.
    pade : Padé approximation.

    Notes
    -----
    At iteration :math:`m` (at which point there are :math:`m` terms in the both the
    numerator and denominator of the approximation), the
    rational approximation in the AAA algorithm takes the barycentric form

    .. math::

        r(z) = n(z)/d(z) =
        \frac{\sum_{j=1}^m\ w_j f_j / (z - z_j)}{\sum_{j=1}^m w_j / (z - z_j)},

    where :math:`z_1,\dots,z_m` are real or complex support points selected from
    `x`, :math:`f_1,\dots,f_m` are the corresponding real or complex data values
    from `y`, and :math:`w_1,\dots,w_m` are real or complex weights.

    Each iteration of the algorithm has two parts: the greedy selection the next support
    point and the computation of the weights. The first part of each iteration is to
    select the next support point to be added :math:`z_{m+1}` from the remaining
    unselected `x`, such that the nonlinear residual
    :math:`|f(z_{m+1}) - n(z_{m+1})/d(z_{m+1})|` is maximised. The algorithm terminates
    when this maximum is less than ``rtol * np.linalg.norm(f, ord=np.inf)``. This means
    the interpolation property is only satisfied up to a tolerance, except at the
    support points where approximation exactly interpolates the supplied data.

    In the second part of each iteration, the weights :math:`w_j` are selected to solve
    the least-squares problem

    .. math::

        \text{minimise}_{w_j}|fd - n| \quad \text{subject to} \quad
        \sum_{j=1}^{m+1} w_j = 1,

    over the unselected elements of `x`.

    One of the challenges with working with rational approximations is the presence of
    Froissart doublets, which are either poles with vanishingly small residues or
    pole-zero pairs that are close enough together to nearly cancel, see [2]_. The
    greedy nature of the AAA algorithm means Froissart doublets are rare. However, if
    `rtol` is set too tight then the approximation will stagnate and many Froissart
    doublets will appear. Froissart doublets can usually be removed by removing support
    points and then resolving the least squares problem. The support point :math:`z_j`,
    which is the closest support point to the pole :math:`a` with residue
    :math:`\alpha`, is removed if the following is satisfied

    .. math::

        |\alpha| / |z_j - a| < \verb|clean_up_tol| \cdot \tilde{f},

    where :math:`\tilde{f}` is the geometric mean of `support_values`.


    References
    ----------
    .. [1] Y. Nakatsukasa, O. Sete, and L. N. Trefethen, "The AAA algorithm for
            rational approximation", SIAM J. Sci. Comp. 40 (2018), A1494-A1522.
            :doi:`10.1137/16M1106122`
    .. [2] J. Gilewicz and M. Pindor, Pade approximants and noise: rational functions,
           J. Comp. Appl. Math. 105 (1999), pp. 285-297.
           :doi:`10.1016/S0377-0427(02)00674-X`

    Examples
    --------

    Here we reproduce a number of the numerical examples from [1]_ as a demonstration
    of the functionality offered by this method.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import AAA
    >>> import warnings

    For the first example we approximate the gamma function on ``[-3.5, 4.5]`` by
    extrapolating from 100 samples in ``[-1.5, 1.5]``.

    >>> from scipy.special import gamma
    >>> sample_points = np.linspace(-1.5, 1.5, num=100)
    >>> r = AAA(sample_points, gamma(sample_points))
    >>> z = np.linspace(-3.5, 4.5, num=1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(z, gamma(z), label="Gamma")
    >>> ax.plot(sample_points, gamma(sample_points), label="Sample points")
    >>> ax.plot(z, r(z).real, '--', label="AAA approximation")
    >>> ax.set(xlabel="z", ylabel="r(z)", ylim=[-8, 8], xlim=[-3.5, 4.5])
    >>> ax.legend()
    >>> plt.show()

    We can also view the poles of the rational approximation and their residues:

    >>> order = np.argsort(r.poles())
    >>> r.poles()[order]
    array([-3.81591039e+00+0.j        , -3.00269049e+00+0.j        ,
           -1.99999988e+00+0.j        , -1.00000000e+00+0.j        ,
            5.85842812e-17+0.j        ,  4.77485458e+00-3.06919376j,
            4.77485458e+00+3.06919376j,  5.29095868e+00-0.97373072j,
            5.29095868e+00+0.97373072j])
    >>> r.residues()[order]
    array([ 0.03658074 +0.j        , -0.16915426 -0.j        ,
            0.49999915 +0.j        , -1.         +0.j        ,
            1.         +0.j        , -0.81132013 -2.30193429j,
           -0.81132013 +2.30193429j,  0.87326839+10.70148546j,
            0.87326839-10.70148546j])

    For the second example, we call `AAA` with a spiral of 1000 points that wind 7.5
    times around the origin in the complex plane.

    >>> z = np.exp(np.linspace(-0.5, 0.5 + 15j*np.pi, 1000))
    >>> r = AAA(z, np.tan(np.pi*z/2), rtol=1e-13)

    We see that AAA takes 12 steps to converge with the following errors:

    >>> r.errors.size
    12
    >>> r.errors
    array([2.49261500e+01, 4.28045609e+01, 1.71346935e+01, 8.65055336e-02,
           1.27106444e-02, 9.90889874e-04, 5.86910543e-05, 1.28735561e-06,
           3.57007424e-08, 6.37007837e-10, 1.67103357e-11, 1.17112299e-13])

    We can also plot the computed poles:

    >>> fig, ax = plt.subplots()
    >>> ax.plot(z.real, z.imag, '.', markersize=2, label="Sample points")
    >>> ax.plot(r.poles().real, r.poles().imag, '.', markersize=5,
    ...         label="Computed poles")
    >>> ax.set(xlim=[-3.5, 3.5], ylim=[-3.5, 3.5], aspect="equal")
    >>> ax.legend()
    >>> plt.show()

    We now demonstrate the removal of Froissart doublets using the `clean_up` method
    using an example from [1]_. Here we approximate the function
    :math:`f(z)=\log(2 + z^4)/(1 + 16z^4)` by sampling it at 1000 roots of unity. The
    algorithm is run with ``rtol=0`` and ``clean_up=False`` to deliberately cause
    Froissart doublets to appear.

    >>> z = np.exp(1j*2*np.pi*np.linspace(0,1, num=1000))
    >>> def f(z):
    ...     return np.log(2 + z**4)/(1 - 16*z**4)
    >>> with warnings.catch_warnings():  # filter convergence warning due to rtol=0
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r = AAA(z, f(z), rtol=0, max_terms=50, clean_up=False)
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> fig, axs = plt.subplots(ncols=2)
    >>> axs[0].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[0].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')

    Now we call the `clean_up` method to remove Froissart doublets.

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r.clean_up()
    4
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> axs[1].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[1].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')
    >>> plt.show()

    The left image shows the poles prior of the approximation ``clean_up=False`` with
    poles with residue less than ``10^-13`` in absolute value shown in red. The right
    image then shows the poles after the `clean_up` method has been called.
    """
    def __init__(self, x, y, *, rtol=None, max_terms=100, clean_up=True,
                 clean_up_tol=1e-13):
        super().__init__(x, y, rtol=rtol, max_terms=max_terms)

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, rtol=None, max_terms=100, clean_up=True,
                          clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        super()._input_validation(x, y)

    @property
    def support_points(self):
        return self._support_points

    @property
    def support_values(self):
        return self._support_values
    
    
    def _compute_weights(self, z, f, rtol, max_terms):
        # Initialization for AAA iteration
        M = np.size(z)
        mask = np.ones(M, dtype=np.bool_)
        dtype = np.result_type(z, f, 1.0)
        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)
        zj = np.empty(max_terms, dtype=dtype)
        fj = np.empty(max_terms, dtype=dtype)
        # Cauchy matrix
        C = np.empty((M, max_terms), dtype=dtype)
        # Loewner matrix
        A = np.empty((M, max_terms), dtype=dtype)
        errors = np.empty(max_terms, dtype=A.real.dtype)
        R = np.repeat(np.mean(f), M)

        # AAA iteration
        for m in range(max_terms):
            # Introduce next support point
            # Select next support point
            jj = np.argmax(np.abs(f[mask] - R[mask]))
            # Update support points
            zj[m] = z[mask][jj]
            # Update data values
            fj[m] = f[mask][jj]
            # Next column of Cauchy matrix
            # Ignore errors as we manually interpolate at support points
            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, m] = 1 / (z - z[mask][jj])
            # Update mask
            mask[np.nonzero(mask)[0][jj]] = False
            # Update Loewner matrix
            # Ignore errors as inf values will be masked out in SVD call
            with np.errstate(invalid="ignore"):
                A[:, m] = (f - fj[m]) * C[:, m]

            # Compute weights
            rows = mask.sum()
            if rows >= m + 1:
                # The usual tall-skinny case
                _, s, V = scipy.linalg.svd(
                    A[mask, : m + 1], full_matrices=False, check_finite=False,
                )
                # Treat case of multiple min singular values
                mm = s == np.min(s)
                # Aim for non-sparse weight vector
                wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
            else:
                # Fewer rows than columns
                V = scipy.linalg.null_space(A[mask, : m + 1], check_finite=False)
                nm = V.shape[-1]
                # Aim for non-sparse wt vector
                wj = V.sum(axis=-1) / np.sqrt(nm)

            # Compute rational approximant
            # Omit columns with `wj == 0`
            i0 = wj != 0
            # Ignore errors as we manually interpolate at support points
            with np.errstate(invalid="ignore"):
                # Numerator
                N = C[:, : m + 1][:, i0] @ (wj[i0] * fj[: m + 1][i0])
                # Denominator
                D = C[:, : m + 1][:, i0] @ wj[i0]
            # Interpolate at support points with `wj !=0`
            D_inf = np.isinf(D) | np.isnan(D)
            D[D_inf] = 1
            N[D_inf] = f[D_inf]
            R = N / D

            # Check if converged
            max_error = np.linalg.norm(f - R, ord=np.inf)
            errors[m] = max_error

            if max_error <= atol:
                break
         
        if m == max_terms - 1:
            warnings.warn(f"AAA failed to converge within {max_terms} iterations.",
                          RuntimeWarning, stacklevel=2)

        # Trim off unused array allocation
        zj = zj[: m + 1]
        fj = fj[: m + 1]

        # Remove support points with zero weight
        i_non_zero = wj != 0
        self.errors = errors[: m + 1]
        self._points = z
        self._values = f
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]

    def clean_up(self, cleanup_tol=1e-13):
        """Automatic removal of Froissart doublets.

        Parameters
        ----------
        cleanup_tol : float, optional
            Poles with residues less than this number times the geometric mean
            of `values` times the minimum distance to `points` are deemed spurious by
            the cleanup procedure, defaults to 1e-13.

        Returns
        -------
        int
            Number of Froissart doublets detected
        """
        # Find negligible residues
        geom_mean_abs_f = scipy.stats.gmean(np.abs(self._values))

        Z_distances = np.min(
            np.abs(np.subtract.outer(self.poles(), self._points)), axis=1
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ii = np.nonzero(
                np.abs(self.residues()) / Z_distances < cleanup_tol * geom_mean_abs_f
            )

        ni = ii[0].size
        if ni == 0:
            return ni

        warnings.warn(f"{ni} Froissart doublets detected.", RuntimeWarning,
                        stacklevel=2)

        # For each spurious pole find and remove closest support point
        closest_spt_point = np.argmin(
            np.abs(np.subtract.outer(self._support_points, self.poles()[ii])), axis=0
        )
        self._support_points = np.delete(self._support_points, closest_spt_point)
        self._support_values = np.delete(self._support_values, closest_spt_point)

        # Remove support points z from sample set
        mask = np.logical_and.reduce(
            np.not_equal.outer(self._points, self._support_points), axis=1
        )
        f = self._values[mask]
        z = self._points[mask]

        # recompute weights, we resolve the least squares problem for the remaining
        # support points

        m = self._support_points.size

        # Cauchy matrix
        C = 1 / np.subtract.outer(z, self._support_points)
        # Loewner matrix
        A = f[:, np.newaxis] * C - C * self._support_values

        # Solve least-squares problem to obtain weights
        _, _, V = scipy.linalg.svd(A, check_finite=False)
        self.weights = np.conj(V[m - 1,:])

        # reset roots, poles, residues as cached values will be wrong with new weights
        self._poles = None
        self._residues = None
        self._roots = None

        return ni


class AAA_filter(_BarycentricRational):
    r"""
    AAA real or complex rational approximation.

    As described in [1]_, the AAA algorithm is a greedy algorithm for approximation by
    rational functions on a real or complex set of points. The rational approximation is
    represented in a barycentric form from which the roots (zeros), poles, and residues
    can be computed.

    Parameters
    ----------
    x : 1D array_like, shape (n,)
        1-D array containing values of the independent variable. Values may be real or
        complex but must be finite.
    y : 1D array_like, shape (n,)
        Function values ``f(x)``. Infinite and NaN values of `values` and
        corresponding values of `points` will be discarded.
    rtol : float, optional
        Relative tolerance, defaults to ``eps**0.75``. If a small subset of the entries
        in `values` are much larger than the rest the default tolerance may be too
        loose. If the tolerance is too tight then the approximation may contain
        Froissart doublets or the algorithm may fail to converge entirely.
    max_terms : int, optional
        Maximum number of terms in the barycentric representation, defaults to ``100``.
        Must be greater than or equal to one.
    clean_up : bool, optional
        Automatic removal of Froissart doublets, defaults to ``True``. See notes for
        more details.
    clean_up_tol : float, optional
        Poles with residues less than this number times the geometric mean
        of `values` times the minimum distance to `points` are deemed spurious by the
        cleanup procedure, defaults to 1e-13. See notes for more details.

    Attributes
    ----------
    support_points : array
        Support points of the approximation. These are a subset of the provided `x` at
        which the approximation strictly interpolates `y`.
        See notes for more details.
    support_values : array
        Value of the approximation at the `support_points`.
    weights : array
        Weights of the barycentric approximation.
    errors : array
        Error :math:`|f(z) - r(z)|_\infty` over `points` in the successive iterations
        of AAA.

    Warns
    -----
    RuntimeWarning
        If `rtol` is not achieved in `max_terms` iterations.

    See Also
    --------
    FloaterHormannInterpolator : Floater-Hormann barycentric rational interpolation.
    pade : Padé approximation.

    Notes
    -----
    At iteration :math:`m` (at which point there are :math:`m` terms in the both the
    numerator and denominator of the approximation), the
    rational approximation in the AAA algorithm takes the barycentric form

    .. math::

        r(z) = n(z)/d(z) =
        \frac{\sum_{j=1}^m\ w_j f_j / (z - z_j)}{\sum_{j=1}^m w_j / (z - z_j)},

    where :math:`z_1,\dots,z_m` are real or complex support points selected from
    `x`, :math:`f_1,\dots,f_m` are the corresponding real or complex data values
    from `y`, and :math:`w_1,\dots,w_m` are real or complex weights.

    Each iteration of the algorithm has two parts: the greedy selection the next support
    point and the computation of the weights. The first part of each iteration is to
    select the next support point to be added :math:`z_{m+1}` from the remaining
    unselected `x`, such that the nonlinear residual
    :math:`|f(z_{m+1}) - n(z_{m+1})/d(z_{m+1})|` is maximised. The algorithm terminates
    when this maximum is less than ``rtol * np.linalg.norm(f, ord=np.inf)``. This means
    the interpolation property is only satisfied up to a tolerance, except at the
    support points where approximation exactly interpolates the supplied data.

    In the second part of each iteration, the weights :math:`w_j` are selected to solve
    the least-squares problem

    .. math::

        \text{minimise}_{w_j}|fd - n| \quad \text{subject to} \quad
        \sum_{j=1}^{m+1} w_j = 1,

    over the unselected elements of `x`.

    One of the challenges with working with rational approximations is the presence of
    Froissart doublets, which are either poles with vanishingly small residues or
    pole-zero pairs that are close enough together to nearly cancel, see [2]_. The
    greedy nature of the AAA algorithm means Froissart doublets are rare. However, if
    `rtol` is set too tight then the approximation will stagnate and many Froissart
    doublets will appear. Froissart doublets can usually be removed by removing support
    points and then resolving the least squares problem. The support point :math:`z_j`,
    which is the closest support point to the pole :math:`a` with residue
    :math:`\alpha`, is removed if the following is satisfied

    .. math::

        |\alpha| / |z_j - a| < \verb|clean_up_tol| \cdot \tilde{f},

    where :math:`\tilde{f}` is the geometric mean of `support_values`.


    References
    ----------
    .. [1] Y. Nakatsukasa, O. Sete, and L. N. Trefethen, "The AAA algorithm for
            rational approximation", SIAM J. Sci. Comp. 40 (2018), A1494-A1522.
            :doi:`10.1137/16M1106122`
    .. [2] J. Gilewicz and M. Pindor, Pade approximants and noise: rational functions,
           J. Comp. Appl. Math. 105 (1999), pp. 285-297.
           :doi:`10.1016/S0377-0427(02)00674-X`

    Examples
    --------

    Here we reproduce a number of the numerical examples from [1]_ as a demonstration
    of the functionality offered by this method.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import AAA
    >>> import warnings

    For the first example we approximate the gamma function on ``[-3.5, 4.5]`` by
    extrapolating from 100 samples in ``[-1.5, 1.5]``.

    >>> from scipy.special import gamma
    >>> sample_points = np.linspace(-1.5, 1.5, num=100)
    >>> r = AAA(sample_points, gamma(sample_points))
    >>> z = np.linspace(-3.5, 4.5, num=1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(z, gamma(z), label="Gamma")
    >>> ax.plot(sample_points, gamma(sample_points), label="Sample points")
    >>> ax.plot(z, r(z).real, '--', label="AAA approximation")
    >>> ax.set(xlabel="z", ylabel="r(z)", ylim=[-8, 8], xlim=[-3.5, 4.5])
    >>> ax.legend()
    >>> plt.show()

    We can also view the poles of the rational approximation and their residues:

    >>> order = np.argsort(r.poles())
    >>> r.poles()[order]
    array([-3.81591039e+00+0.j        , -3.00269049e+00+0.j        ,
           -1.99999988e+00+0.j        , -1.00000000e+00+0.j        ,
            5.85842812e-17+0.j        ,  4.77485458e+00-3.06919376j,
            4.77485458e+00+3.06919376j,  5.29095868e+00-0.97373072j,
            5.29095868e+00+0.97373072j])
    >>> r.residues()[order]
    array([ 0.03658074 +0.j        , -0.16915426 -0.j        ,
            0.49999915 +0.j        , -1.         +0.j        ,
            1.         +0.j        , -0.81132013 -2.30193429j,
           -0.81132013 +2.30193429j,  0.87326839+10.70148546j,
            0.87326839-10.70148546j])

    For the second example, we call `AAA` with a spiral of 1000 points that wind 7.5
    times around the origin in the complex plane.

    >>> z = np.exp(np.linspace(-0.5, 0.5 + 15j*np.pi, 1000))
    >>> r = AAA(z, np.tan(np.pi*z/2), rtol=1e-13)

    We see that AAA takes 12 steps to converge with the following errors:

    >>> r.errors.size
    12
    >>> r.errors
    array([2.49261500e+01, 4.28045609e+01, 1.71346935e+01, 8.65055336e-02,
           1.27106444e-02, 9.90889874e-04, 5.86910543e-05, 1.28735561e-06,
           3.57007424e-08, 6.37007837e-10, 1.67103357e-11, 1.17112299e-13])

    We can also plot the computed poles:

    >>> fig, ax = plt.subplots()
    >>> ax.plot(z.real, z.imag, '.', markersize=2, label="Sample points")
    >>> ax.plot(r.poles().real, r.poles().imag, '.', markersize=5,
    ...         label="Computed poles")
    >>> ax.set(xlim=[-3.5, 3.5], ylim=[-3.5, 3.5], aspect="equal")
    >>> ax.legend()
    >>> plt.show()

    We now demonstrate the removal of Froissart doublets using the `clean_up` method
    using an example from [1]_. Here we approximate the function
    :math:`f(z)=\log(2 + z^4)/(1 + 16z^4)` by sampling it at 1000 roots of unity. The
    algorithm is run with ``rtol=0`` and ``clean_up=False`` to deliberately cause
    Froissart doublets to appear.

    >>> z = np.exp(1j*2*np.pi*np.linspace(0,1, num=1000))
    >>> def f(z):
    ...     return np.log(2 + z**4)/(1 - 16*z**4)
    >>> with warnings.catch_warnings():  # filter convergence warning due to rtol=0
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r = AAA(z, f(z), rtol=0, max_terms=50, clean_up=False)
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> fig, axs = plt.subplots(ncols=2)
    >>> axs[0].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[0].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')

    Now we call the `clean_up` method to remove Froissart doublets.

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r.clean_up()
    4
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> axs[1].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[1].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')
    >>> plt.show()

    The left image shows the poles prior of the approximation ``clean_up=False`` with
    poles with residue less than ``10^-13`` in absolute value shown in red. The right
    image then shows the poles after the `clean_up` method has been called.
    """
    def __init__(self, x, y, *, rtol=None, max_terms=100, clean_up=True,
                 clean_up_tol=1e-13):
        super().__init__(x, y, rtol=rtol, max_terms=max_terms)

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, rtol=None, max_terms=100, clean_up=True,
                          clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        super()._input_validation(x, y)

    @property
    def support_points(self):
        return self._support_points

    @property
    def support_values(self):
        return self._support_values
    
    
    def _compute_weights(self, z, f, rtol, max_terms):
        # Initialization for AAA iteration
        M = np.size(z)
        mask = np.ones(M, dtype=np.bool_)
        dtype = np.result_type(z, f, 1.0)
        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)
        zj = np.empty(max_terms, dtype=dtype)
        fj = np.empty(max_terms, dtype=dtype)
        # Cauchy matrix
        C = np.empty((M, max_terms), dtype=dtype)
        # Loewner matrix
        A = np.empty((M, max_terms), dtype=dtype)
        errors = np.empty(max_terms, dtype=A.real.dtype)
        R = np.repeat(np.mean(f), M)

        def _compute_roots2(z, f, w):
            B = np.eye(len(w) + 1)
            B[0,0] = 0
            E = np.block([[0, w],
                  [f[:,None], np.diag(z)]])
            pol = scipy.linalg.eigvals(E, B)
            return pol[np.isfinite(pol)]
        
        # AAA iteration
        for m in range(max_terms):
            # Introduce next support point
            # Select next support point
            jj = np.argmax(np.abs(f[mask] - R[mask]))
            # Update support points
            zj[m] = z[mask][jj]
            # Update data values
            fj[m] = f[mask][jj]
            # Next column of Cauchy matrix
            # Ignore errors as we manually interpolate at support points
            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, m] = 1 / (z - z[mask][jj])
            # Update mask
            mask[np.nonzero(mask)[0][jj]] = False
            # Update Loewner matrix
            # Ignore errors as inf values will be masked out in SVD call
            with np.errstate(invalid="ignore"):
                A[:, m] = (f - fj[m]) * C[:, m]

            # Compute weights
            rows = mask.sum()
            if rows >= m + 1:
                # The usual tall-skinny case
                _, s, V = scipy.linalg.svd(
                    A[mask, : m + 1], full_matrices=False, check_finite=False,
                )
                # Treat case of multiple min singular values
                mm = s == np.min(s)
                # Aim for non-sparse weight vector
                wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
            else:
                # Fewer rows than columns
                V = scipy.linalg.null_space(A[mask, : m + 1], check_finite=False)
                nm = V.shape[-1]
                # Aim for non-sparse wt vector
                wj = V.sum(axis=-1) / np.sqrt(nm)

            # Compute rational approximant
            # Omit columns with `wj == 0`
            i0 = wj != 0
            # Ignore errors as we manually interpolate at support points
            with np.errstate(invalid="ignore"):
                # Numerator
                N = C[:, : m + 1][:, i0] @ (wj[i0] * fj[: m + 1][i0])
                # Denominator
                D = C[:, : m + 1][:, i0] @ wj[i0]
            # Interpolate at support points with `wj !=0`
            D_inf = np.isinf(D) | np.isnan(D)
            D[D_inf] = 1
            N[D_inf] = f[D_inf]
            R = N / D

            # Check if converged
            max_error = np.linalg.norm(f - R, ord=np.inf)
            errors[m] = max_error

            zj_temp  = zj[: m + 1]
            fj_temp  = fj[: m + 1]
            i_non_zero = wj != 0
            
            final_poles = _compute_roots2(zj_temp[i_non_zero], np.ones_like(fj_temp[i_non_zero]), wj[i_non_zero])
            if max_error <= atol and np.min(np.abs(final_poles.imag +1e-9*1j)) > 1e-3:
                print("Found suitable pole-free interpolation \n")
                print("The poles are",final_poles)
                print("This is how it passes critirion", np.min(np.abs(final_poles.imag +1e-9*1j)))
                break
         
        if m == max_terms - 1 and np.min(np.abs(final_poles.imag +1e-9*1j)) > 1e-3:
            print(f"Filtered AAA failed to converge within {max_terms} iterations. The last pred is still real pole-free")
        elif m == max_terms - 1 and np.min(np.abs(final_poles.imag +1e-9*1j)) < 1e-3:
            print(f"Filtered AAA failed to converge within {max_terms} iterations. The last pred is not real pole-free")

        # Trim off unused array allocation
        zj = zj[: m + 1]
        fj = fj[: m + 1]

        # Remove support points with zero weight
        i_non_zero = wj != 0
        self.errors = errors[: m + 1]
        self._points = z
        self._values = f
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]

    def clean_up(self, cleanup_tol=1e-13):
        """Automatic removal of Froissart doublets.

        Parameters
        ----------
        cleanup_tol : float, optional
            Poles with residues less than this number times the geometric mean
            of `values` times the minimum distance to `points` are deemed spurious by
            the cleanup procedure, defaults to 1e-13.

        Returns
        -------
        int
            Number of Froissart doublets detected
        """
        # Find negligible residues
        geom_mean_abs_f = scipy.stats.gmean(np.abs(self._values))

        Z_distances = np.min(
            np.abs(np.subtract.outer(self.poles(), self._points)), axis=1
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ii = np.nonzero(
                np.abs(self.residues()) / Z_distances < cleanup_tol * geom_mean_abs_f
            )

        ni = ii[0].size
        if ni == 0:
            return ni

        warnings.warn(f"{ni} Froissart doublets detected.", RuntimeWarning,
                        stacklevel=2)

        # For each spurious pole find and remove closest support point
        closest_spt_point = np.argmin(
            np.abs(np.subtract.outer(self._support_points, self.poles()[ii])), axis=0
        )
        self._support_points = np.delete(self._support_points, closest_spt_point)
        self._support_values = np.delete(self._support_values, closest_spt_point)

        # Remove support points z from sample set
        mask = np.logical_and.reduce(
            np.not_equal.outer(self._points, self._support_points), axis=1
        )
        f = self._values[mask]
        z = self._points[mask]

        # recompute weights, we resolve the least squares problem for the remaining
        # support points

        m = self._support_points.size

        # Cauchy matrix
        C = 1 / np.subtract.outer(z, self._support_points)
        # Loewner matrix
        A = f[:, np.newaxis] * C - C * self._support_values

        # Solve least-squares problem to obtain weights
        _, _, V = scipy.linalg.svd(A, check_finite=False)
        self.weights = np.conj(V[m - 1,:])

        # reset roots, poles, residues as cached values will be wrong with new weights
        self._poles = None
        self._residues = None
        self._roots = None

        return ni


class AAA_adding0(_BarycentricRational):
    r"""
    AAA real or complex rational approximation, but with a seeded first support point:
    always include the sample point z_j with minimal |z_j| (closest to 0) as the first
    interpolation/support point.

    Parameters are the same as AAA.
    """
    def __init__(self, x, y, *, rtol=None, max_terms=100, clean_up=True,
                 clean_up_tol=1e-13):
        super().__init__(x, y, rtol=rtol, max_terms=max_terms)

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, rtol=None, max_terms=100, clean_up=True,
                          clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        super()._input_validation(x, y)

    @property
    def support_points(self):
        return self._support_points

    @property
    def support_values(self):
        return self._support_values

    def _compute_weights(self, z, f, rtol, max_terms):
        # Initialization for AAA iteration
        M = np.size(z)
        mask = np.ones(M, dtype=np.bool_)
        dtype = np.result_type(z, f, 1.0)

        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)

        zj = np.empty(max_terms, dtype=dtype)
        fj = np.empty(max_terms, dtype=dtype)

        # Cauchy matrix
        C = np.empty((M, max_terms), dtype=dtype)
        # Loewner matrix
        A = np.empty((M, max_terms), dtype=dtype)

        errors = np.empty(max_terms, dtype=np.result_type(f.real, 1.0))

        def _compute_roots2(z, f, w):
            B = np.eye(len(w) + 1)
            B[0,0] = 0
            E = np.block([[0, w],
                  [f[:,None], np.diag(z)]])
            pol = scipy.linalg.eigvals(E, B)
            pol = pol[np.isfinite(pol)]
            return pol

        # ---------- NEW: seed the first support point as argmin |z| ----------
        j0 = int(np.argmin(np.abs(z)))   # index in the full array
        zj[0] = z[j0]
        fj[0] = f[j0]
        mask[j0] = False

        with np.errstate(divide="ignore", invalid="ignore"):
            C[:, 0] = 1 / (z - zj[0])
        with np.errstate(invalid="ignore"):
            A[:, 0] = (f - fj[0]) * C[:, 0]

        # Start with constant approx; will be overwritten immediately
        R = np.repeat(np.mean(f), M)

        # Compute weights + approximant for m = 0
        m = 0
        rows = mask.sum()
        if rows >= m + 1:
            _, s, V = scipy.linalg.svd(
                A[mask, :m + 1], full_matrices=False, check_finite=False
            )
            mm = (s == np.min(s))
            wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
        else:
            V = scipy.linalg.null_space(A[mask, :m + 1], check_finite=False)
            nm = V.shape[-1]
            wj = V.sum(axis=-1) / np.sqrt(nm)

        i0 = (wj != 0)
        with np.errstate(invalid="ignore"):
            N = C[:, :m + 1][:, i0] @ (wj[i0] * fj[:m + 1][i0])
            D = C[:, :m + 1][:, i0] @ wj[i0]

        D_inf = np.isinf(D) | np.isnan(D)
        D[D_inf] = 1
        N[D_inf] = f[D_inf]
        R = N / D

        max_error = np.linalg.norm(f - R, ord=np.inf)
        errors[m] = max_error

        if max_error <= atol:
            # converged already (rare but possible)
            pass
        # -------------------------------------------------------------------

        # Continue AAA iterations from m = 1
        for m in range(1, max_terms):
            # Select next support point from remaining nodes
            jj = np.argmax(np.abs(f[mask] - R[mask]))
            zj[m] = z[mask][jj]
            fj[m] = f[mask][jj]

            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, m] = 1 / (z - zj[m])

            # Update mask (map masked-index jj back to global index)
            mask[np.nonzero(mask)[0][jj]] = False

            with np.errstate(invalid="ignore"):
                A[:, m] = (f - fj[m]) * C[:, m]

            # Compute weights
            rows = mask.sum()
            if rows >= m + 1:
                _, s, V = scipy.linalg.svd(
                    A[mask, :m + 1], full_matrices=False, check_finite=False
                )
                mm = (s == np.min(s))
                wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
            else:
                V = scipy.linalg.null_space(A[mask, :m + 1], check_finite=False)
                nm = V.shape[-1]
                wj = V.sum(axis=-1) / np.sqrt(nm)

            # Compute rational approximant
            i0 = (wj != 0)
            with np.errstate(invalid="ignore"):
                N = C[:, :m + 1][:, i0] @ (wj[i0] * fj[:m + 1][i0])
                D = C[:, :m + 1][:, i0] @ wj[i0]

            D_inf = np.isinf(D) | np.isnan(D)
            D[D_inf] = 1
            N[D_inf] = f[D_inf]
            R = N / D

            # Check if converged
            max_error = np.linalg.norm(f - R, ord=np.inf)
            errors[m] = max_error

            zj_temp  = zj[: m + 1]
            fj_temp  = fj[: m + 1]
            i_non_zero = wj != 0
            
            final_poles = _compute_roots2(zj_temp[i_non_zero], np.ones_like(fj_temp[i_non_zero]), wj[i_non_zero])
            if max_error <= atol and np.min(np.abs(final_poles.imag)) > 5*1e-4:
                # print("closest pole to real axis:", np.min(np.abs(final_poles.imag)), "\n")
                break
        if np.min(np.abs(final_poles.imag)) < 5*1e-4:
            # print("Cannot find suitable pole-free interpolation, Return naive AAA \n")
            print("real part of the pole:", np.abs(final_poles.real), "\n")
            
        
        # if m == max_terms - 1 and np.min(np.abs(final_poles.imag)) > 1e-4:
        #     # print(f"Filtered AAA failed to converge within {max_terms} iterations. return pred: Real pole-free")
        # elif m == max_terms - 1 and np.min(np.abs(final_poles.imag)) < 1e-4:
            # print(f"Filtered AAA failed to converge within {max_terms} iterations. return pred: Not real pole-free")
       
        # Trim off unused array allocation
        zj = zj[:m + 1]
        fj = fj[:m + 1]

        # Remove support points with zero weight
        i_non_zero = (wj != 0)
        self.errors = errors[:m + 1]
        self._points = z
        self._values = f
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]

    def clean_up(self, cleanup_tol=1e-13):
        """Automatic removal of Froissart doublets (same as AAA)."""
        geom_mean_abs_f = scipy.stats.gmean(np.abs(self._values))

        Z_distances = np.min(
            np.abs(np.subtract.outer(self.poles(), self._points)), axis=1
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ii = np.nonzero(
                np.abs(self.residues()) / Z_distances < cleanup_tol * geom_mean_abs_f
            )

        ni = ii[0].size
        if ni == 0:
            return ni

        warnings.warn(f"{ni} Froissart doublets detected.", RuntimeWarning,
                      stacklevel=2)

        closest_spt_point = np.argmin(
            np.abs(np.subtract.outer(self._support_points, self.poles()[ii])), axis=0
        )
        self._support_points = np.delete(self._support_points, closest_spt_point)
        self._support_values = np.delete(self._support_values, closest_spt_point)

        mask = np.logical_and.reduce(
            np.not_equal.outer(self._points, self._support_points), axis=1
        )
        f = self._values[mask]
        z = self._points[mask]

        m = self._support_points.size
        C = 1 / np.subtract.outer(z, self._support_points)
        A = f[:, np.newaxis] * C - C * self._support_values

        _, _, V = scipy.linalg.svd(A, check_finite=False)
        self.weights = np.conj(V[m - 1, :])

        self._poles = None
        self._residues = None
        self._roots = None

        return ni


class AAA_holo_interp(AAA_adding0):
    r"""
    Modified AAA interpolation seeded at the sample point nearest zero.

    This variant follows :class:`AAA_adding0`, but accepts convergence only when the
    interpolation error is below tolerance and no computed pole has real part in the
    forbidden interval specified by ``pole_real_window``. If the pole condition fails,
    it retries up to ``max_trials`` times using a fresh random perturbation of the
    original function values. If ``perturbation_magnitude`` is not set, the
    perturbation magnitude defaults to the first failed run's final error. If a full
    batch of ``max_trials`` perturbations fails, the perturbation magnitude is
    multiplied by 5 and another batch is attempted.
    """

    def __init__(self, x, y, *, rtol=None, max_terms=100,
                 pole_real_window=(0.1, 0.6), max_trials=10,
                 perturbation_magnitude=None, random_state=None,
                 clean_up=True, clean_up_tol=1e-13):
        pole_real_window = self._normalize_pole_real_window(pole_real_window)
        self._pole_real_window = pole_real_window
        _BarycentricRational.__init__(
            self, x, y, rtol=rtol, max_terms=max_terms,
            pole_real_window=pole_real_window, max_trials=max_trials,
            perturbation_magnitude=perturbation_magnitude,
            random_state=random_state
        )

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, rtol=None, max_terms=100,
                          pole_real_window=(0.1, 0.6), max_trials=10,
                          perturbation_magnitude=None, random_state=None,
                          clean_up=True, clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        max_trials = operator.index(max_trials)
        if max_trials < 1:
            raise ValueError("`max_trials` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        self._normalize_pole_real_window(pole_real_window)
        self._normalize_perturbation_magnitude(perturbation_magnitude)
        _BarycentricRational._input_validation(self, x, y)

    @staticmethod
    def _normalize_pole_real_window(pole_real_window):
        if len(pole_real_window) != 2:
            raise ValueError("`pole_real_window` must contain exactly two values.")
        lo, hi = pole_real_window
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("`pole_real_window` values must be finite.")
        if lo > hi:
            raise ValueError("`pole_real_window` must be ordered as (low, high).")
        return lo, hi

    @staticmethod
    def _normalize_perturbation_magnitude(perturbation_magnitude):
        if perturbation_magnitude is None:
            return None
        perturbation_magnitude = float(perturbation_magnitude)
        if not np.isfinite(perturbation_magnitude):
            raise ValueError("`perturbation_magnitude` must be finite.")
        if perturbation_magnitude < 0:
            raise ValueError("`perturbation_magnitude` must be non-negative.")
        return perturbation_magnitude

    @staticmethod
    def _compute_poles_from_weights(z, w):
        B = np.eye(len(w) + 1)
        B[0, 0] = 0
        E = np.zeros_like(B, dtype=np.result_type(z, w, 1.0))
        E[0, 1:] = w
        E[1:, 0] = 1
        np.fill_diagonal(E[1:, 1:], z)
        poles = scipy.linalg.eigvals(E, B)
        return poles[np.isfinite(poles)]

    def _poles_outside_real_window(self, poles):
        lo, hi = self._pole_real_window
        return not np.any((poles.real >= lo) & (poles.real <= hi))

    def _compute_weights(self, z, f, rtol, max_terms, pole_real_window,
                         max_trials, perturbation_magnitude, random_state):
        M = np.size(z)
        max_trials = operator.index(max_trials)
        if max_trials < 1:
            raise ValueError("`max_trials` must be an integer value greater than or "
                             "equal to one.")
        dtype = np.result_type(z, f, 1.0)
        real_dtype = np.result_type(f.real, 1.0)
        rng = np.random.default_rng(random_state)
        perturbation_magnitude = self._normalize_perturbation_magnitude(
            perturbation_magnitude
        )

        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)
        f_original = f.copy()

        def _random_perturbation(scale):
            scale = float(abs(scale))
            if not np.isfinite(scale) or scale == 0:
                scale = np.finfo(real_dtype).eps * max(1.0, np.linalg.norm(f, ord=np.inf))

            noise = rng.standard_normal(M)
            if np.issubdtype(dtype, np.complexfloating):
                noise = (noise + 1j * rng.standard_normal(M)) / np.sqrt(2)

            noise_norm = np.linalg.norm(noise, ord=np.inf)
            if noise_norm == 0:
                return np.zeros(M, dtype=dtype)
            return (scale * noise / noise_norm).astype(dtype, copy=False)

        def _run_aaa(f_work, target_error):
            mask = np.ones(M, dtype=np.bool_)
            zj = np.empty(max_terms, dtype=dtype)
            fj = np.empty(max_terms, dtype=dtype)

            # Cauchy matrix
            C = np.empty((M, max_terms), dtype=dtype)
            # Loewner matrix
            A = np.empty((M, max_terms), dtype=dtype)

            errors = np.empty(max_terms, dtype=real_dtype)

            # Seed the first support point as argmin |z|.
            j0 = int(np.argmin(np.abs(z)))
            zj[0] = z[j0]
            fj[0] = f_work[j0]
            mask[j0] = False

            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, 0] = 1 / (z - zj[0])
            with np.errstate(invalid="ignore"):
                A[:, 0] = (f_work - fj[0]) * C[:, 0]

            R = np.repeat(np.mean(f_work), M)
            wj = np.ones(1, dtype=dtype)
            final_poles = np.empty(0, dtype=dtype)
            max_error = np.inf

            for m in range(max_terms):
                if m > 0:
                    # Select next support point from remaining nodes.
                    jj = np.argmax(np.abs(f_work[mask] - R[mask]))
                    zj[m] = z[mask][jj]
                    fj[m] = f_work[mask][jj]

                    with np.errstate(divide="ignore", invalid="ignore"):
                        C[:, m] = 1 / (z - zj[m])

                    # Map masked-index jj back to global index.
                    mask[np.nonzero(mask)[0][jj]] = False

                    with np.errstate(invalid="ignore"):
                        A[:, m] = (f_work - fj[m]) * C[:, m]

                # Compute weights
                rows = mask.sum()
                if rows >= m + 1:
                    _, s, V = scipy.linalg.svd(
                        A[mask, :m + 1], full_matrices=False, check_finite=False
                    )
                    mm = (s == np.min(s))
                    wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
                else:
                    V = scipy.linalg.null_space(A[mask, :m + 1], check_finite=False)
                    nm = V.shape[-1]
                    wj = V.sum(axis=-1) / np.sqrt(nm)

                # Compute rational approximant.
                i0 = (wj != 0)
                with np.errstate(invalid="ignore"):
                    N = C[:, :m + 1][:, i0] @ (wj[i0] * fj[:m + 1][i0])
                    D = C[:, :m + 1][:, i0] @ wj[i0]

                D_inf = np.isinf(D) | np.isnan(D)
                D[D_inf] = 1
                N[D_inf] = f_work[D_inf]
                R = N / D

                max_error = np.linalg.norm(f_work - R, ord=np.inf)
                errors[m] = max_error

                zj_temp = zj[:m + 1]
                fj_temp = fj[:m + 1]
                i_non_zero = (wj != 0)
                final_poles = self._compute_poles_from_weights(
                    zj_temp[i_non_zero], wj[i_non_zero]
                )

                if max_error <= target_error:
                    break

            return zj, fj, wj, errors, m, max_error, final_poles, f_work

        target_error = atol
        retried_with_perturbation = False
        zj, fj, wj, errors, m, max_error, final_poles, f_fit = _run_aaa(
            f_original, target_error
        )

        if not self._poles_outside_real_window(final_poles):
            perturbation_scale = (
                max_error if perturbation_magnitude is None
                else perturbation_magnitude
            )
            if not np.isfinite(perturbation_scale) or perturbation_scale == 0:
                perturbation_scale = (
                    np.finfo(real_dtype).eps * max(1.0, np.linalg.norm(f, ord=np.inf))
                )
            retried_with_perturbation = True
            total_trials = 0

            while not self._poles_outside_real_window(final_poles):
                for _ in range(max_trials):
                    # Each trial is a fresh perturbation of the original data, not a
                    # cumulative perturbation of the previous trial.
                    f_trial = f_original + _random_perturbation(perturbation_scale)
                    zj, fj, wj, errors, m, max_error, final_poles, f_fit = _run_aaa(
                        f_trial, target_error
                    )
                    total_trials += 1
                    if self._poles_outside_real_window(final_poles):
                        break

                if self._poles_outside_real_window(final_poles):
                    break

                perturbation_scale *= 2

        if retried_with_perturbation:
            print("AAA_holo_interp final max_error:", max_error)
            print("AAA_holo_interp final target_error:", target_error)
            print("AAA_holo_interp perturbation_magnitude:", perturbation_scale)
            print("AAA_holo_interp perturbation_trials:", total_trials)
            print("AAA_holo_interp accepted a perturbed interpolation")

        # Trim off unused array allocation.
        zj = zj[:m + 1]
        fj = fj[:m + 1]

        # Remove support points with zero weight.
        i_non_zero = (wj != 0)
        self.errors = errors[:m + 1]
        self._points = z
        self._values = f_fit
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]


class AAA_holo_interp_v2(AAA_holo_interp):
    r"""
    Modified AAA interpolation with original-data error tracking.

    This variant is the same as :class:`AAA_holo_interp`, except that the AAA
    iteration error is measured against the original unperturbed function values:
    ``np.linalg.norm(f_original - R, ord=np.inf)``. Perturbation trials still fit
    perturbed data, but convergence is judged by the fit to the original data.
    """

    def _compute_weights(self, z, f, rtol, max_terms, pole_real_window,
                         max_trials, perturbation_magnitude, random_state):
        M = np.size(z)
        max_trials = operator.index(max_trials)
        if max_trials < 1:
            raise ValueError("`max_trials` must be an integer value greater than or "
                             "equal to one.")
        dtype = np.result_type(z, f, 1.0)
        real_dtype = np.result_type(f.real, 1.0)
        rng = np.random.default_rng(random_state)
        perturbation_magnitude = self._normalize_perturbation_magnitude(
            perturbation_magnitude
        )

        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)
        f_original = f.copy()

        def _random_perturbation(scale):
            scale = float(abs(scale))
            if not np.isfinite(scale) or scale == 0:
                scale = np.finfo(real_dtype).eps * max(
                    1.0, np.linalg.norm(f_original, ord=np.inf)
                )

            noise = rng.standard_normal(M)
            if np.issubdtype(dtype, np.complexfloating):
                noise = (noise + 1j * rng.standard_normal(M)) / np.sqrt(2)

            noise_norm = np.linalg.norm(noise, ord=np.inf)
            if noise_norm == 0:
                return np.zeros(M, dtype=dtype)
            return (scale * noise / noise_norm).astype(dtype, copy=False)

        def _run_aaa(f_work, target_error):
            mask = np.ones(M, dtype=np.bool_)
            zj = np.empty(max_terms, dtype=dtype)
            fj = np.empty(max_terms, dtype=dtype)

            # Cauchy matrix
            C = np.empty((M, max_terms), dtype=dtype)
            # Loewner matrix
            A = np.empty((M, max_terms), dtype=dtype)

            errors = np.empty(max_terms, dtype=real_dtype)

            # Seed the first support point as argmin |z|.
            j0 = int(np.argmin(np.abs(z)))
            zj[0] = z[j0]
            fj[0] = f_work[j0]
            mask[j0] = False

            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, 0] = 1 / (z - zj[0])
            with np.errstate(invalid="ignore"):
                A[:, 0] = (f_work - fj[0]) * C[:, 0]

            R = np.repeat(np.mean(f_work), M)
            wj = np.ones(1, dtype=dtype)
            final_poles = np.empty(0, dtype=dtype)
            max_error = np.inf

            for m in range(max_terms):
                if m > 0:
                    # Select next support point from remaining nodes.
                    jj = np.argmax(np.abs(f_work[mask] - R[mask]))
                    zj[m] = z[mask][jj]
                    fj[m] = f_work[mask][jj]

                    with np.errstate(divide="ignore", invalid="ignore"):
                        C[:, m] = 1 / (z - zj[m])

                    # Map masked-index jj back to global index.
                    mask[np.nonzero(mask)[0][jj]] = False

                    with np.errstate(invalid="ignore"):
                        A[:, m] = (f_work - fj[m]) * C[:, m]

                # Compute weights
                rows = mask.sum()
                if rows >= m + 1:
                    _, s, V = scipy.linalg.svd(
                        A[mask, :m + 1], full_matrices=False, check_finite=False
                    )
                    mm = (s == np.min(s))
                    wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
                else:
                    V = scipy.linalg.null_space(A[mask, :m + 1], check_finite=False)
                    nm = V.shape[-1]
                    wj = V.sum(axis=-1) / np.sqrt(nm)

                # Compute rational approximant.
                i0 = (wj != 0)
                with np.errstate(invalid="ignore"):
                    N = C[:, :m + 1][:, i0] @ (wj[i0] * fj[:m + 1][i0])
                    D = C[:, :m + 1][:, i0] @ wj[i0]

                D_inf = np.isinf(D) | np.isnan(D)
                D[D_inf] = 1
                N[D_inf] = f_work[D_inf]
                R = N / D

                max_error = np.linalg.norm(f_original - R, ord=np.inf)
                errors[m] = max_error

                zj_temp = zj[:m + 1]
                fj_temp = fj[:m + 1]
                i_non_zero = (wj != 0)
                final_poles = self._compute_poles_from_weights(
                    zj_temp[i_non_zero], wj[i_non_zero]
                )

                if max_error <= target_error:
                    break

            return zj, fj, wj, errors, m, max_error, final_poles, f_work

        target_error = atol
        retried_with_perturbation = False
        zj, fj, wj, errors, m, max_error, final_poles, f_fit = _run_aaa(
            f_original, target_error
        )

        def _accepted(error, poles):
            return error <= target_error and self._poles_outside_real_window(poles)

        if not _accepted(max_error, final_poles):
            perturbation_scale = (
                max_error if perturbation_magnitude is None
                else perturbation_magnitude
            )
            if not np.isfinite(perturbation_scale) or perturbation_scale == 0:
                perturbation_scale = (
                    np.finfo(real_dtype).eps
                    * max(1.0, np.linalg.norm(f_original, ord=np.inf))
                )
            retried_with_perturbation = True
            total_trials = 0

            while not _accepted(max_error, final_poles):
                for _ in range(max_trials):
                    # Each trial is a fresh perturbation of the original data, not a
                    # cumulative perturbation of the previous trial.
                    f_trial = f_original + _random_perturbation(perturbation_scale)
                    zj, fj, wj, errors, m, max_error, final_poles, f_fit = _run_aaa(
                        f_trial, target_error
                    )
                    total_trials += 1
                    if _accepted(max_error, final_poles):
                        break

                if _accepted(max_error, final_poles):
                    break

                perturbation_scale *= 10
                print("AAA_holo_interp_v2 trying perturbation scale:", perturbation_scale)

        if retried_with_perturbation:
            print("AAA_holo_interp_v2 final max_error:", max_error)
            print("AAA_holo_interp_v2 final target_error:", target_error)
            print("AAA_holo_interp_v2 perturbation_magnitude:", perturbation_scale)
            print("AAA_holo_interp_v2 perturbation_trials:", total_trials)
            print("AAA_holo_interp_v2 accepted a perturbed interpolation")

        # Trim off unused array allocation.
        zj = zj[:m + 1]
        fj = fj[:m + 1]

        # Remove support points with zero weight.
        i_non_zero = (wj != 0)
        self.errors = errors[:m + 1]
        self._points = z
        self._values = f_fit
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]


class AAA_tube(AAA_adding0):
    r"""
    AAA interpolation constrained by a pole-free complex tube.

    This variant follows :class:`AAA_adding0`, but accepts convergence only when:

    1. the AAA interpolation error is below tolerance,
    2. no computed pole lies inside the delta tube around the real interval covered
       by the sample points, and
    3. the interpolant has modulus less than one on ``Ns`` sampled boundary points
       of that tube.

    The tube is the stadium-shaped region around ``tube_real_window``:
    a rectangle ``lo <= real(z) <= hi, |imag(z)| <= delta`` plus semicircular caps of
    radius ``delta`` at ``lo`` and ``hi``. If ``tube_real_window`` is not supplied,
    the interval defaults to ``[min(real(x)), max(real(x))]``.
    """

    def __init__(self, x, y, *, delta, Ns=100, rtol=None, max_terms=100,
                 tube_real_window=None, clean_up=False, clean_up_tol=1e-13):
        self._tube_delta = self._normalize_delta(delta)
        self._tube_Ns = self._normalize_Ns(Ns)
        self._tube_real_window = self._normalize_tube_real_window(tube_real_window)
        _BarycentricRational.__init__(
            self, x, y, delta=self._tube_delta, Ns=self._tube_Ns,
            rtol=rtol, max_terms=max_terms,
            tube_real_window=self._tube_real_window,
        )

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, delta, Ns=100, rtol=None, max_terms=100,
                          tube_real_window=None, clean_up=False, clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        self._normalize_delta(delta)
        self._normalize_Ns(Ns)
        self._normalize_tube_real_window(tube_real_window)
        _BarycentricRational._input_validation(self, x, y)

    @staticmethod
    def _normalize_delta(delta):
        delta = float(delta)
        if not np.isfinite(delta):
            raise ValueError("`delta` must be finite.")
        if delta <= 0:
            raise ValueError("`delta` must be positive.")
        return delta

    @staticmethod
    def _normalize_Ns(Ns):
        Ns = operator.index(Ns)
        if Ns < 4:
            raise ValueError("`Ns` must be an integer value greater than or equal to 4.")
        return Ns

    @staticmethod
    def _normalize_tube_real_window(tube_real_window):
        if tube_real_window is None:
            return None
        if len(tube_real_window) != 2:
            raise ValueError("`tube_real_window` must contain exactly two values.")
        lo, hi = tube_real_window
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("`tube_real_window` values must be finite.")
        if lo > hi:
            raise ValueError("`tube_real_window` must be ordered as (low, high).")
        return lo, hi

    @staticmethod
    def _compute_poles_from_weights(z, w):
        B = np.eye(len(w) + 1)
        B[0, 0] = 0
        E = np.zeros_like(B, dtype=np.result_type(z, w, 1.0))
        E[0, 1:] = w
        E[1:, 0] = 1
        np.fill_diagonal(E[1:, 1:], z)
        poles = scipy.linalg.eigvals(E, B)
        return poles[np.isfinite(poles)]

    @staticmethod
    def _tube_boundary_points(lo, hi, delta, Ns):
        if np.isclose(lo, hi):
            theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
            return lo + delta * np.exp(1j * theta)

        n_upper = max(2, Ns // 4)
        n_right = max(2, Ns // 4)
        n_lower = max(2, Ns // 4)
        n_left = max(2, Ns - n_upper - n_right - n_lower)

        upper = np.linspace(lo, hi, n_upper, endpoint=False) + 1j * delta
        theta_right = np.linspace(np.pi / 2, -np.pi / 2, n_right, endpoint=False)
        right_cap = hi + delta * np.exp(1j * theta_right)
        lower = np.linspace(hi, lo, n_lower, endpoint=False) - 1j * delta
        theta_left = np.linspace(-np.pi / 2, np.pi / 2, n_left, endpoint=False)
        left_cap = lo + delta * np.exp(1j * theta_left)
        return np.concatenate([upper, right_cap, lower, left_cap])

    @staticmethod
    def _poles_inside_tube(poles, lo, hi, delta):
        if poles.size == 0:
            return np.zeros(0, dtype=bool)

        real_part = poles.real
        imag_abs = np.abs(poles.imag)

        in_strip = (real_part >= lo) & (real_part <= hi) & (imag_abs <= delta)
        left_cap = np.abs(poles - lo) <= delta
        right_cap = np.abs(poles - hi) <= delta

        return in_strip | left_cap | right_cap

    @classmethod
    def _poles_outside_tube(cls, poles, lo, hi, delta):
        return not np.any(cls._poles_inside_tube(poles, lo, hi, delta))

    @staticmethod
    def _evaluate_barycentric(z_eval, support_points, support_values, weights):
        z_eval = np.asarray(z_eval)
        zv = np.ravel(z_eval)

        with np.errstate(invalid="ignore", divide="ignore"):
            C = 1 / np.subtract.outer(zv, support_points)
            r = C @ (weights * support_values) / (C @ weights)

        ii = np.nonzero(np.isnan(r))[0]
        for jj in ii:
            if np.isnan(zv[jj]) or not np.any(zv[jj] == support_points):
                continue
            r[jj] = support_values[zv[jj] == support_points].squeeze()

        return np.reshape(r, z_eval.shape)

    def _compute_weights(self, z, f, delta, Ns, rtol, max_terms, tube_real_window):
        M = np.size(z)
        mask = np.ones(M, dtype=np.bool_)
        dtype = np.result_type(z, f, 1.0)
        real_dtype = np.result_type(f.real, 1.0)

        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)

        if tube_real_window is None:
            tube_lo = float(np.min(np.real(z)))
            tube_hi = float(np.max(np.real(z)))
        else:
            tube_lo, tube_hi = tube_real_window
        boundary_points = self._tube_boundary_points(tube_lo, tube_hi, delta, Ns)

        zj = np.empty(max_terms, dtype=dtype)
        fj = np.empty(max_terms, dtype=dtype)

        # Cauchy matrix
        C = np.empty((M, max_terms), dtype=dtype)
        # Loewner matrix
        A = np.empty((M, max_terms), dtype=dtype)
        errors = np.empty(max_terms, dtype=real_dtype)

        # Seed the first support point as argmin |z|.
        j0 = int(np.argmin(np.abs(z)))
        zj[0] = z[j0]
        fj[0] = f[j0]
        mask[j0] = False

        with np.errstate(divide="ignore", invalid="ignore"):
            C[:, 0] = 1 / (z - zj[0])
        with np.errstate(invalid="ignore"):
            A[:, 0] = (f - fj[0]) * C[:, 0]

        R = np.repeat(np.mean(f), M)
        wj = np.ones(1, dtype=dtype)
        final_poles = np.empty(0, dtype=dtype)
        boundary_max = np.inf
        best_valid = None

        for m in range(max_terms):
            if m > 0:
                # Select next support point from remaining nodes.
                jj = np.argmax(np.abs(f[mask] - R[mask]))
                zj[m] = z[mask][jj]
                fj[m] = f[mask][jj]

                with np.errstate(divide="ignore", invalid="ignore"):
                    C[:, m] = 1 / (z - zj[m])

                # Map masked-index jj back to global index.
                mask[np.nonzero(mask)[0][jj]] = False

                with np.errstate(invalid="ignore"):
                    A[:, m] = (f - fj[m]) * C[:, m]

            # Compute weights
            rows = mask.sum()
            if rows >= m + 1:
                _, s, V = scipy.linalg.svd(
                    A[mask, :m + 1], full_matrices=False, check_finite=False
                )
                mm = (s == np.min(s))
                wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
            else:
                V = scipy.linalg.null_space(A[mask, :m + 1], check_finite=False)
                nm = V.shape[-1]
                wj = V.sum(axis=-1) / np.sqrt(nm)

            i0 = (wj != 0)
            with np.errstate(invalid="ignore"):
                N = C[:, :m + 1][:, i0] @ (wj[i0] * fj[:m + 1][i0])
                D = C[:, :m + 1][:, i0] @ wj[i0]

            D_inf = np.isinf(D) | np.isnan(D)
            D[D_inf] = 1
            N[D_inf] = f[D_inf]
            R = N / D

            max_error = np.linalg.norm(f - R, ord=np.inf)
            errors[m] = max_error

            zj_temp = zj[:m + 1]
            fj_temp = fj[:m + 1]
            i_non_zero = (wj != 0)
            support_points = zj_temp[i_non_zero]
            support_values = fj_temp[i_non_zero]
            weights = wj[i_non_zero]

            final_poles = self._compute_poles_from_weights(support_points, weights)
            boundary_values = self._evaluate_barycentric(
                boundary_points, support_points, support_values, weights
            )
            boundary_max = np.linalg.norm(boundary_values, ord=np.inf)
            pole_ok = self._poles_outside_tube(final_poles, tube_lo, tube_hi, delta)
            boundary_ok = np.isfinite(boundary_max) and boundary_max < 1

            if pole_ok and boundary_ok:
                if best_valid is None or max_error < best_valid[5]:
                    best_valid = (
                        m,
                        zj[:m + 1].copy(),
                        fj[:m + 1].copy(),
                        wj.copy(),
                        errors[:m + 1].copy(),
                        max_error,
                        final_poles.copy(),
                        boundary_max,
                    )

            if (
                max_error <= atol
                and pole_ok
                and boundary_ok
            ):
                break

        final_ok = (
            max_error <= atol
            and self._poles_outside_tube(final_poles, tube_lo, tube_hi, delta)
            and np.isfinite(boundary_max)
            and boundary_max < 1
        )
        if not final_ok and best_valid is not None:
            (
                m,
                zj,
                fj,
                wj,
                valid_errors,
                max_error,
                final_poles,
                boundary_max,
            ) = best_valid
            errors[:m + 1] = valid_errors
            print("AAA_tube using best tube-safe iterate before max_terms.")

        inside_tube_mask = self._poles_inside_tube(final_poles, tube_lo, tube_hi, delta)
        no_tube_safe_iterate = (
            np.any(inside_tube_mask)
            or not np.isfinite(boundary_max)
            or boundary_max >= 1
        )
        if not (
            max_error <= atol
            and self._poles_outside_tube(final_poles, tube_lo, tube_hi, delta)
            and np.isfinite(boundary_max)
            and boundary_max < 1
        ):
            warnings.warn(
                "AAA_tube failed to satisfy all constraints within "
                f"{max_terms} iterations.",
                RuntimeWarning,
                stacklevel=2,
            )
            if no_tube_safe_iterate:
                print(
                    "AAA_tube warning: could not find an interpolation with poles "
                    "outside the delta tube and boundary max < 1."
                )
                print("AAA_tube warning tube real interval:", (tube_lo, tube_hi))
                print("AAA_tube warning tube delta:", delta)
                print("AAA_tube warning final poles:", final_poles)
                print("AAA_tube warning poles inside tube:", final_poles[inside_tube_mask])
                print("AAA_tube warning maximum boundary |r(z)|:", boundary_max)

        print("AAA_tube final poles:", final_poles)
        print("AAA_tube tube real interval:", (tube_lo, tube_hi))
        print("AAA_tube tube delta:", delta)
        print("AAA_tube poles inside tube:", final_poles[inside_tube_mask])
        print("AAA_tube maximum boundary |r(z)|:", boundary_max)

        # Trim off unused array allocation.
        zj = zj[:m + 1]
        fj = fj[:m + 1]

        # Remove support points with zero weight.
        i_non_zero = (wj != 0)
        self.errors = errors[:m + 1]
        self._points = z
        self._values = f
        self._tube_boundary_points_used = boundary_points
        self._tube_boundary_max = boundary_max
        self._tube_poles = final_poles
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]

class FloaterHormannInterpolator(_BarycentricRational):
    r"""
    Floater-Hormann barycentric rational interpolation.

    As described in [1]_, the method of Floater and Hormann computes weights for a
    Barycentric rational interpolant with no poles on the real axis.

    Parameters
    ----------
    x : 1D array_like, shape (n,)
        1-D array containing values of the independent variable. Values may be real or
        complex but must be finite.
    y : array_like, shape (n, ...)
        Array containing values of the dependent variable. Infinite and NaN values
        of `values` and corresponding values of `x` will be discarded.
    d : int, optional
        Blends ``n - d`` degree `d` polynomials together. For ``d = n - 1`` it is
        equivalent to polynomial interpolation. Must satisfy ``0 <= d < n``,
        defaults to 3.

    Attributes
    ----------
    weights : array
        Weights of the barycentric approximation.

    See Also
    --------
    AAA : Barycentric rational approximation of real and complex functions.
    pade : Padé approximation.

    Notes
    -----
    The Floater-Hormann interpolant is a rational function that interpolates the data
    with approximation order :math:`O(h^{d+1})`. The rational function blends ``n - d``
    polynomials of degree `d` together to produce a rational interpolant that contains
    no poles on the real axis, unlike `AAA`. The interpolant is given
    by

    .. math::

        r(x) = \frac{\sum_{i=0}^{n-d} \lambda_i(x) p_i(x)}
        {\sum_{i=0}^{n-d} \lambda_i(x)},

    where :math:`p_i(x)` is an interpolating polynomials of at most degree `d` through
    the points :math:`(x_i,y_i),\dots,(x_{i+d},y_{i+d}), and :math:`\lambda_i(z)` are
    blending functions defined by

    .. math::

        \lambda_i(x) = \frac{(-1)^i}{(x - x_i)\cdots(x - x_{i+d})}.

    When ``d = n - 1`` this reduces to polynomial interpolation.

    Due to its stability following barycentric representation of the above equation
    is used instead for computation

    .. math::

        r(z) = \frac{\sum_{k=1}^m\ w_k f_k / (x - x_k)}{\sum_{k=1}^m w_k / (x - x_k)},

    where the weights :math:`w_j` are computed as

    .. math::

        w_k &= (-1)^{k - d} \sum_{i \in J_k} \prod_{j = i, j \neq k}^{i + d}
        1/|x_k - x_j|, \\
        J_k &= \{ i \in I: k - d \leq i \leq k\},\\
        I &= \{0, 1, \dots, n - d\}.

    References
    ----------
    .. [1] M.S. Floater and K. Hormann, "Barycentric rational interpolation with no
           poles and high rates of approximation", Numer. Math. 107, 315 (2007).
           :doi:`10.1007/s00211-007-0093-y`

    Examples
    --------

    Here we compare the method against polynomial interpolation for an example where
    the polynomial interpolation fails due to Runge's phenomenon.

    >>> import numpy as np
    >>> from scipy.interpolate import (FloaterHormannInterpolator,
    ...                                BarycentricInterpolator)
    >>> def f(z):
    ...     return 1/(1 + z**2)
    >>> z = np.linspace(-5, 5, num=15)
    >>> r = FloaterHormannInterpolator(z, f(z))
    >>> p = BarycentricInterpolator(z, f(z))
    >>> zz = np.linspace(-5, 5, num=1000)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot(zz, r(zz), label="Floater=Hormann")
    >>> ax.plot(zz, p(zz), label="Polynomial")
    >>> ax.legend()
    >>> plt.show()
    """
    def __init__(self, points, values, *, d=3):
        super().__init__(points, values, d=d)

    def _input_validation(self, x, y, d):
        d = operator.index(d)
        if not (0 <= d < len(x)):
            raise ValueError("`d` must satisfy 0 <= d < n")

        super()._input_validation(x, y)

    def _compute_weights(self, z, f, d):
        # Floater and Hormann 2007 Eqn. (18) 3 equations later
        w = np.zeros_like(z, dtype=np.result_type(z, 1.0))
        n = w.size
        for k in range(n):
            for i in range(max(k-d, 0), min(k+1, n-d)):
                w[k] += 1/np.prod(np.abs(np.delete(z[k] - z[i : i + d + 1], k - i)))
        w *= (-1.)**(np.arange(n) - d)

        return z, f, w
