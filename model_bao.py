import numpy as np


class PowerBAOModel(object):

    def __init__(self, pklin, pksmooth):
      self.pklin_interp = pklin
      self.pksmooth_interp = pksmooth

    def wiggles_smooth(self, k):
        return self.pklin_interp(k) / self.pksmooth_interp(k)

    def wiggles_damped_iso(self, k, sigmanl=0.):
        return 1. + (self.wiggles_smooth(k) - 1.) * np.exp(-1. / 2. * (sigmanl * k)**2)

    def wiggles_damped_aniso(self, k, mu, sigmapar=0., sigmaper=0.):
        return 1. + (self.wiggles_smooth(k) - 1.) * np.exp(-1. / 2. * ((sigmapar * k * mu)**2 + (sigmaper * k)**2 * (1. - mu**2)))

    @staticmethod
    def polynomial(k, ap0=0., ap1=0., ap2=0., ap3=0., am1=0., am2=0., am3=0., am4=0., **kwarg):
        return ap0 + ap1 * k + ap2 * k**2 + ap3 * k**3 + am1 / k + am2 / k**2 + am3 / k**3 + am4 / k**4

    def DFoG(self, k, mu=1., sigmav=1.):
        return (1. + (k * mu * sigmav)**2 / 2.)**(-2)

    def pk_smooth_iso(self, k, b=1., sigmas=0., **kwargs):
        DFoG = self.DFoG(k, mu=1., sigmav=abs(sigmas))
        if sigmas < 0: DFoG = 1. / DFoG
        return b**2 * self.pksmooth_interp(k) * DFoG

    def pk_galaxy_iso(self, kobs, qiso=1., sigmanl=0., decoupled=True, **kwargs):
        k = kobs if decoupled else kobs / qiso
        Psm = self.pk_smooth_iso(k,**kwargs)
        return Psm * self.wiggles_damped_iso(kobs / qiso, sigmanl=sigmanl)

    def pk_smooth_aniso(self, k, mu, b=1., beta=0., sigmapar=0., sigmaper=0., sigmas=0., sigmasm=0., recon=False, kobs=None, decoupled=False, **kwargs):
        if decoupled: k = kobs
        if recon: r = 1. - np.exp(-1. / 2 * (k * sigmasm)**2)
        else: r = 1.
        DFoG = self.DFoG(k, mu, sigmav=abs(sigmas))
        if sigmas < 0: DFoG = 1. / DFoG
        return b**2 * (1. + beta * mu**2 * r)**2 * self.pksmooth_interp(k) * DFoG

    def pk_galaxy_aniso(self, k, mu, sigmapar=0., sigmaper=0., **kwargs):
        Psm = self.pk_smooth_aniso(k, mu, sigmapar=sigmapar, sigmaper=sigmaper, **kwargs)
        return Psm * self.wiggles_damped_aniso(k, mu, sigmapar=sigmapar, sigmaper=sigmaper)



def enforce_shape(x, y, grid=True):
    """
    Broadcast ``x`` and ``y`` arrays.

    Parameters
    ----------
    x : array_like
        Input x-array, scalar, 1D or 2D if not ``grid``.
    y : array_like
        Input y-array, scalar, 1D or 2D if not ``grid``.
    grid : bool, default=True
        If ``grid``, and ``x``, ``y`` not scalars, add dimension to ``x`` such that ``x`` and ``y`` can be broadcast
        (e.g. ``x*y``, is shape ``(len(x),) + y.shape``).
        Else, simply return x,y.

    Returns
    -------
    x : array
        x-array.
    y : array
        y-array.
    """
    x, y = np.asarray(x), np.asarray(y)
    if (not grid) or (x.ndim == 0) or (y.ndim == 0):
        return x, y
    return x[:,None], y


class AnisotropicScaling(object):

    def __init__(self, input_model):
        self.input_model = input_model

    def set_scaling(self, qpar=1, qperp=1):
        self.qpar = qpar
        self.qperp = qperp
        self.qap = qpar / qperp
        self.qiso = (self.qperp**2 * self.qpar)**(1. / 3.)

    def kmu_scaling(self, k, mu, grid=True):
        factor_ap = np.sqrt(1 + mu**2 * (1. / self.qap**2 - 1))
        k, mu = enforce_shape(k, mu, grid=grid)
        # Beutler 2016 (arXiv: 1607.03150) eq 44
        kap = k / self.qperp * factor_ap
        # Beutler 2016 (arXiv: 1607.03150) eq 45
        muap = mu / self.qap / factor_ap
        return kap, muap

    def smu_scaling(self, s, mu, grid=True):
        factor_ap = np.sqrt(mu**2 * (self.qap**2 - 1) + 1)
        # Hou 2018 (arXiv: 2007.08998) eq 8
        sap = s * self.qperp * factor_ap
        muap = mu * self.qap / factor_ap
        return sap, muap

    def pk_mu(self, x, mu=0., grid=True, qpar=1., qperp=1., **kwargs):
        self.set_scaling(qpar=qpar, qperp=qperp)
        kap, muap = self.kmu_scaling(x, mu, grid=grid)
        return 1. / self.qiso**3 * self.input_model(kap, mu=muap, **kwargs)

    def xi_mu(self, x, mu=0., grid=True, qpar=1., qperp=1., **kwargs):
        self.set_scaling(qpar=qpar, qperp=qperp)
        sap, muap = self.smu_scaling(x, mu, grid=grid)
        return self.input_model(sap, mu=muap, **kwargs)


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])/2.


class BaseMultipoleIntegration(object):

    """Base class to perform integration over Legendre polynomials."""

    def __init__(self, mu=100, ells=(0,2,4), sym=False):
        r"""
        Initialize :class:`BaseMultipoleIntegration`.

        Parameters
        ----------
        mu : int, array, default=100
            :math:`\mu` cosine angle to integrate on.
            :math:`\mu`-coordinate array or number of :math:`\mu`-bins.

        ells : tuple
            Multipole orders.

        sym : bool, default=False
            Whether integration is symmetric around :math:`\mu = 0`.
            In this case, and if input ``mu`` is the number of bins, only integrates between 0 and 1.
        """
        self.mu = mu
        if np.ndim(self.mu) == 0:
            self.mu = np.linspace(0. if sym else -1., 1., self.mu + 1)
        self.ells = ells
        self.set_mu_weights()

    def __call__(self, array):
        """Integrate input array."""
        return np.sum(array * self.weights[:, None, :], axis=-1)


# TODO: implement gauss-legendre integration
class TrapzMultipoleIntegration(BaseMultipoleIntegration):

    """Class performing trapzeoidal integration over Legendre polynomials."""

    def set_mu_weights(self):
        """Set weights for trapezoidal integration."""
        muw_trapz = weights_trapz(self.mu)
        from scipy import special
        self.weights = np.array([muw_trapz * (2 * ell + 1.) * special.legendre(ell)(self.mu) for ell in self.ells]) / (self.mu[-1] - self.mu[0])
