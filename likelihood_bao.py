import os
import re
import logging

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import BOSS
from cosmoprimo import Cosmology, PowerSpectrumBAOFilter
from pycorr import setup_logging

from model_bao import PowerBAOModel, AnisotropicScaling, TrapzMultipoleIntegration
import environment
from environment import Measurement


base_dir = './results'

zeffs = np.linspace(0.1, 1.9, 10)

logger = logging.getLogger('PowerBAOLikelihood')


class BaseLikelihood(object):

    def list_to_dict(self, args):
        d = dict(zip(self.varied, args))
        d.update(self.params['optional'])
        return d

    def dict_to_list(self, **d):
        return [d[par] for par in self.varied]

    def chi2(self, args):
        model = self.model(args)
        diff = (self.data - model).flatten()
        return diff.dot(self.invcov).T.dot(diff)

    def loglkl(self, args):
        return -0.5 * self.chi2(args)

    def logposterior(self, args):
        logprior = self.logprior(args)
        if not np.isfinite(logprior): return logprior
        return self.loglkl(args) + logprior

    def logprior(self, args):
        for b,v in zip(self.params['bounds'], args):
            if not (b[0] <= v <= b[1]):
                return -np.inf
        return 0.

    @property
    def varied(self):
        return self.params['varied']

    @property
    def ndata(self):
        return sum(map(len, self.data))

    @property
    def nvaried(self):
        return len(self.varied)

    def start(self, seed=None):
        rng = np.random.RandomState(seed=seed)
        return [rng.normal(np.mean(b), (b[1] - b[0]) / 100.) for b in self.params['bounds']]


class BaseBAOLikelihood(BaseLikelihood):

    def __init__(self, path_data, xlim=(0.02,0.3), iso=False):
        self.iso = iso
        self.set_data_covariance(path_data, xlim=xlim)
        self.set_model()

    def set_data_covariance(self, path_data, xlim=None):
        data = Measurement.load(path_data)
        self.zeff = data.attrs['zeff']
        self.recon = data.attrs.get('recon', False)
        self.ells = (0,) if self.iso else (0, 2)
        self.xdata, self.data, self._mask = [], [], []
        for ill,ell in enumerate(self.ells):
            mask = (data.x > xlim[0]) & (data.x < xlim[-1])
            self.xdata.append(data.x[mask])
            self.data.append(data.y[ill, mask])
            self._mask.append(mask)
        self.xdata = self.xdata[0] # same x for all ells, just to simplify
        self.data = np.array(self.data)

        mask = np.concatenate(self._mask)
        self.cov = data.cov[np.ix_(mask, mask)]
        self.invcov = np.linalg.inv(self.cov)
        assert np.allclose(self.cov.dot(self.invcov), np.eye(*self.cov.shape))
        self.stddev = np.diag(self.cov)**0.5
        self.stddev = np.array([self.stddev[ill * len(self.xdata):(ill+1) * len(self.xdata)] for ill, ell in enumerate(self.ells)])


class PowerBAOLikelihood(BaseBAOLikelihood):

    def set_model(self, cosmo=None):
        self.cosmo = cosmo or BOSS(engine='class')
        pklin = self.cosmo.get_fourier().pk_interpolator().to_1d(self.zeff)
        pksmooth = PowerSpectrumBAOFilter(pklin, engine='wallish2018').smooth_pk_interpolator()
        model = PowerBAOModel(pklin, pksmooth)
        self.params = {}

        if self.iso:
            bb = ['m1', 'p0', 'p1', 'p2']
            self.params['varied'] = ['qiso', 'b'] + ['a{}'.format(i) for i in bb]
            self.params['bounds'] = [(0.8, 1.2), (0.5, 4.0)] + [(-20, 20)]*len(bb)
            if not self.recon:
                self.params['optional'] = {'sigmanl': (1./3. * 9.4**2 + 2./3. * 4.8**2)**0.5}
            else:
                self.params['optional'] = {'sigmanl': (1./3. * 7.0**2 + 2./3. * 2.0**2)**0.5}
            template_bb = re.compile('a[p,m][0-9]')

            def pk(k=None, **params):
                if k is None: k = self.xdata
                pk = model.pk_galaxy_iso(k, **params)
                bb = [par for par in params if re.match(template_bb, par)]
                pk += PowerBAOModel.polynomial(k, **{par: params[par] * 1e3 for par in bb})
                return pk[None,:]

        else:
            bb = ['m1', 'p0', 'p1']
            self.params['varied'] = ['qpar', 'qperp', 'b', 'beta']
            self.params['bounds'] = [(0.8, 1.2), (0.8, 1.2), (0.5, 4.0), (0., 2)]
            if not self.recon:
                self.params['optional'] = {'sigmapar': 9.4, 'sigmaper': 4.8}
            else:
                self.params['optional'] = {'sigmaper': 7.0, 'sigmaper': 2.0}
            self.params['optional']['recon'] = self.recon
            for ell in self.ells:
                self.params['varied'] += ['a{}_l{:d}'.format(i, ell) for i in bb]
                self.params['bounds'] += [(-20, 20)] * len(bb)
            pk_mu = AnisotropicScaling(model.pk_galaxy_aniso).pk_mu
            integration = TrapzMultipoleIntegration(ells=self.ells, sym=True)
            template_bb = {ell: 'a[p,m][0-9]_l{:d}'.format(ell) for ell in self.ells}

            def pk(k=None, **params):
                if k is None: k = self.xdata
                pk = pk_mu(k, integration.mu, **params)
                pk = integration(pk)
                for ill,ell in enumerate(self.ells):
                    old = [par for par in params if re.match(template_bb[ell], par)]
                    new = [par.replace('_l{:d}'.format(ell), '') for par in old]
                    nparams = {npar: params[opar] * 1e3 for opar,npar in zip(old, new)}
                    pk[ill] += PowerBAOModel.polynomial(k, **nparams)
                return pk

        self._pk_model = pk

    def model(self, args, k=None):
        params = self.list_to_dict(args)
        return self._pk_model(k=k, **params)

    def plot(self, params, path_figure=None):
        k = np.linspace(self.xdata[0], self.xdata[-1],100)
        model = self.model(params, k=k)
        ax = plt.gca()
        for ill, ell in enumerate(self.ells):
            ax.errorbar(self.xdata, self.xdata * self.data[ill], self.xdata * self.stddev[ill], linestyle='none', color='C{:d}'.format(ill))
            ax.plot(k, k * model[ill], color='C{:d}'.format(ill), label='$\ell = {:d}$'.format(ell))

        ax.legend(loc=1, fontsize=15)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.set_xlabel('$k$ [$h / \\mathrm{Mpc}$]', fontsize=17)
        ax.set_ylabel('$k P(k)$ [$(\\mathrm{{Mpc}} / h)^{2}$]', fontsize=17)

        if path_figure is not None:
            logger.info('Saving figure to {}.'.format(path_figure))
            plt.savefig(path_figure, bbox_inches='tight', pad_inches=0.1,dpi=200)
            plt.close(plt.gcf())


def minimize(likelihood):
    from scipy import optimize
    result = optimize.minimize(likelihood.chi2, likelihood.start(seed=42), method='SLSQP', bounds=likelihood.params['bounds'])
    logger.info(result.message)
    msg = 'Reduced chi2 is {:.4f}/({:d} - {:d}) = {:.4f}'.format(result.fun, likelihood.ndata, likelihood.nvaried, result.fun / (likelihood.ndata - likelihood.nvaried))
    logger.info(msg)
    assert result.success
    return result.x


def sample(likelihood, nsteps=2000, path_samples=None):
    import emcee
    ndim = len(likelihood.varied)
    nwalkers = 2 * ndim
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.logposterior)
    start = [likelihood.start(seed=seed) for seed in range(nwalkers)]
    sampler.run_mcmc(start, nsteps)
    samples = sampler.get_chain(flat=True)
    if path_samples is not None:
        logger.info('Saving samples to {}.'.format(path_samples))
        np.save(path_samples, samples)
    return samples


def plot_samples(likelihood, samples, path_figure=None):
    import corner
    fig = corner.corner(samples, labels=likelihood.varied, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={'fontsize':14})
    if path_figure is not None:
        logger.info('Saving figure to {}.'.format(path_figure))
        fig.savefig(path_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close(fig)


def save_measurement(likelihood, mean, cov, path_measurement):
    if likelihood.iso:
        params = ['qiso']
    else:
        params = ['qpar', 'qperp']
    indices = [likelihood.varied.index(par) for par in params]
    mean = mean[indices]
    cov = cov[np.ix_(indices, indices)]
    m = Measurement(x=likelihood.zeff, y=mean, cov=cov, attrs={'iso': likelihood.iso})
    m.save(path_measurement)


def path_measurement(space='power', tracer='LRGpCMASS', iso=True):
    return os.path.join(base_dir, 'bao_eBOSS_{}_{}_{}.npy'.format(tracer, space, 'iso' if iso else 'aniso'))


def path_sim_measurement(space='power', tracer='LRGpCMASS', z=0.5, iso=True):
    return os.path.join(base_dir, 'bao_sim_{}_{}_{}_{:.2f}.npy'.format(tracer, space, 'iso' if iso else 'aniso',z))


def list_path_sim_measurement(**kwargs):
    return [path_sim_measurement(z=z, **kwargs) for z in zeffs]


if __name__ == '__main__':

    setup_logging()

    environment.mkdir(base_dir)

    #data = 'eBOSS'
    data = 'sims'
    tracer = 'LRGpCMASS'
    recon = False
    iso = True
    xlim = (0.02, 0.3)
    run = ['minimize', 'mcmc']
    siso = 'iso' if iso else 'aniso'
    space = 'power'

    def run_pipeline(path):

        if 'minimize' in run:
            x = minimize(likelihood)
            d = likelihood.list_to_dict(x)
            if likelihood.iso:
                msg = 'alpha_iso is {:.4f}'.format(d['qiso'])
            else:
                msg = 'alpha_par/perp are {:.4f}/{:.4f}'.format(d['qpar'], d['qperp'])
            logger.info(msg)
            likelihood.plot(x, path_figure=path_figure)

        if 'mcmc' in run:
            samples = sample(likelihood, nsteps=2000, path_samples=path_samples)
            samples = np.load(path_samples)
            samples = samples[len(samples) // 2:]
            x = np.median(samples, axis=0)
            cov = np.cov(samples.T, ddof=1)
            likelihood.plot(x, path_figure=path_figure)
            plot_samples(likelihood, samples, path_figure_samples)
            save_measurement(likelihood, x, cov, path)

    if data == 'eBOSS':
        path_data = environment.path_measurement(space=space, tracer=tracer, recon=recon)
        #path_data = estimators.path_measurement(space=space, tracer=tracer, recon=recon)
        path_figure = os.path.join(base_dir,'fit_bao_{}_{}_{}_{}.png'.format(data, tracer, space, siso))
        path_samples = os.path.join(base_dir,'samples_bao_{}_{}_{}_{}.npy'.format(data, tracer, space, siso))
        path_figure_samples = os.path.join(base_dir,'samples_{}_{}_bao_{}_{}.png'.format(data, tracer, space, siso))
        likelihood = PowerBAOLikelihood(path_data, xlim=xlim, iso=iso)
        run_pipeline(path_measurement(tracer=tracer, iso=iso))

    if data == 'sims':
        for z in zeffs:
            path_data = environment.path_sim_measurement(space=space, tracer=tracer, z=z)
            path_figure = os.path.join(base_dir, 'fit_bao_{}_{}_{}_{}_{:.2f}.png'.format(data, tracer, space, siso, z))
            path_samples = os.path.join(base_dir, 'samples_bao_{}_{}_{}_{}_{:.2f}.npy'.format(data, tracer, space, siso, z))
            path_figure_samples = os.path.join(base_dir, 'samples_{}_{}_bao_{}_{}_{:.2f}.png'.format(data, tracer, space, siso, z))
            likelihood = PowerBAOLikelihood(path_data, xlim=xlim, iso=iso)
            run_pipeline(path_sim_measurement(tracer=tracer, iso=iso, z=z))
