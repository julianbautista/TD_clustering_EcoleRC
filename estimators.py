import os
import logging

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

from cosmoprimo.fiducial import BOSS
from pypower import CatalogFFTPower, PowerSpectrumStatistics, setup_logging
from pycorr import TwoPointCorrelationFunction

import environment
from environment import Measurement


logger = logging.getLogger('Analysis')

base_dir = './results'


def read_catalog(path, zcuts=None, downsample=2.):
    cosmo = BOSS(engine='class')
    P0 = 1e4
    catalog = Table.read(path)
    if downsample < 1.:
        rng = np.random.RandomState(seed=42)
        mask = rng.uniform(0., 1., len(catalog)) < downsample
        catalog = catalog[mask]
    if zcuts is not None:
        catalog = catalog[(catalog['Z'] > zcuts[0]) & (catalog['Z'] < zcuts[1])]
    positions = [catalog['RA'], catalog['DEC'], cosmo.comoving_radial_distance(catalog['Z'])]
    wfkp = 1. / (1. + catalog['NZ'] * P0)
    weights = catalog['WEIGHT_SYSTOT'] * catalog['WEIGHT_CP'] * catalog['WEIGHT_NOZ'] * catalog['WEIGHT_FKP']
    return positions, weights


def estimate_correlation_function(path_data, path_randoms, path_result, path_shifted=None, nthreads=4, zcuts=None, downsample_randoms=2., recon=False):
    data_positions, data_weights = read_catalog(path_data, zcuts=zcuts)
    randoms_positions, randoms_weights = read_catalog(path_randoms, zcuts=zcuts, downsample=downsample_randoms)
    shifted_positions, shifted_weights = None, None
    if path_shifted is not None:
        shifted_positions, shifted_weights = read_catalog(path_shifted, zcuts=zcuts, downsample=downsample_randoms)

    correlation = TwoPointCorrelationFunction(mode='smu', edges=(np.linspace(0., 140, 71), np.linspace(-1., 1., 100)),
                                              data_positions1=data_positions, data_weights1=data_weights,
                                              randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                                              shifted_positions1=shifted_positions, shifted_weights1=shifted_weights,
                                              position_type='rdd', nthreads=nthreads, dtype='f8')
    correlation.save(path_result)


def plot_correlation_function(path_results, labels=None, linestyles='-', path_figure=None):
    if isinstance(path_results, str):
        path_results = [path_results]
        labels = [labels]
        linestyles = [linestyles]

    ax = plt.gca()
    for path_result, label, linestyle in zip(path_results, labels, linestyles):
        data = Measurement.load(path_result)
        for ill, ell in enumerate(data.attrs['ells']):
            label_ = '$\ell = {:d}$'.format(ell)
            if ill == 0 and label is not None:
                label_ = '{} {}'.format(label, label_)
            ax.plot(data.x, data.x**2 * data.y[ill], label=label_, linestyle=linestyle, color='C{:d}'.format(ill))

    ax.legend(loc=1, fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    ax.set_xlabel('$s$ [$\\mathrm{Mpc} / h$]', fontsize=17)
    ax.set_ylabel('$s^{2} \\xi(s)$ [$(\\mathrm{{Mpc}} / h)^{2}$]', fontsize=17)
    if path_figure is not None:
        logger.info('Saving figure to {}.'.format(path_figure))
        plt.savefig(path_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)


def estimate_power_spectrum(path_data, path_randoms, path_result, path_shifted=None, zcuts=None, downsample_randoms=2.):

    data_positions, data_weights = read_catalog(path_data, zcuts=zcuts)
    randoms_positions, randoms_weights = read_catalog(path_randoms, zcuts=zcuts, downsample=downsample_randoms)
    shifted_positions, shifted_weights = None, None
    if path_shifted is not None:
        shifted_positions, shifted_weights = read_catalog(path_shifted, zcuts=zcuts, downsample=downsample_randoms)

    power = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights,
                            randoms_positions1=randoms_positions, randoms_weights1=randoms_weights,
                            shifted_positions1=shifted_positions, shifted_weights1=shifted_weights,
                            boxsize=5000., nmesh=256, interlacing=3,
                            edges={'step': 0.005}, ells=(0, 2, 4), position_type='rdd', dtype='f8', mpiroot=0).poles
    power.save(path_result)


def plot_power_spectrum(path_results, labels=None, linestyles='-', path_figure=None):

    if isinstance(path_results, str):
        path_results = [path_results]
        labels = [labels]
        linestyles = [linestyles]

    ax = plt.gca()
    for path_result, label, linestyle in zip(path_results, labels, linestyles):
        data = Measurement.load(path_result)
        for ill, ell in enumerate(data.attrs['ells']):
            label_ = '$\ell = {:d}$'.format(ell)
            if ill == 0 and label is not None:
                label_ = '{} {}'.format(label,label_)
            ax.plot(data.x, data.x * data.y[ill], label=label_, linestyle=linestyle, color='C{:d}'.format(ill))

    ax.legend(loc=1, fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    ax.set_xlabel('$k$ [$h / \\mathrm{Mpc}$]', fontsize=17)
    ax.set_ylabel('$k P(k)$ [$(\\mathrm{{Mpc}} / h)^{2}$]', fontsize=17)
    if path_figure is not None:
        logger.info('Saving figure to {}.'.format(path_figure))
        plt.savefig(path_figure, bbox_inches='tight', pad_inches=0.1, dpi=200)


def average_correlation_function(path_results, effective_area, path_average, cov=None, attrs=None):
    # Combination based on area, not the best choice, but...
    attrs = attrs or {}
    ells = (0, 2, 4)
    x, y = 0, 0
    for path_result, area in zip(path_results, effective_area):
        s, xi = TwoPointCorrelationFunction.load(path_result)(ells=ells, return_sep=True)
        x += s * area
        y += xi * area
    x /= sum(effective_area)
    y /= sum(effective_area)
    m = Measurement(x, y, cov=cov, attrs={**attrs, **{'space':'correlation', 'ells':ells}})
    m.save(path_average)


def average_power_spectrum(path_results, effective_area, path_average, cov=None, attrs=None):
    attrs = attrs or {}
    y = 0
    for path_result,area in zip(path_results, effective_area):
        power = PowerSpectrumStatistics.load(path_result)
        ells = power.ells
        x, pk = power(return_k=True, complex=False)
        y += pk * area
    y /= sum(effective_area)
    m = Measurement(x, y, cov=cov, attrs={**attrs, **{'space': 'power', 'ells': ells}})
    m.save(path_average)


if __name__ == '__main__':

    setup_logging()

    environment.mkdir(base_dir)
    tracer = 'LRGpCMASS'
    run = ['estimator', 'average', 'plot'][2:]
    space = 'correlation'
    #space = 'power'
    recon = False
    #recon = True

    if 'estimator' in run:
        path_results = []
        for cap in environment.caps:
            path_data, path_randoms = environment.path_catalogs(tracer=tracer, cap=cap, recon=recon)
            if recon:
                path_shifted = path_randoms
                path_randoms = environment.path_catalogs(tracer=tracer, cap=cap, recon=False)[1]
            path_result = environment.path_measurement(space=space, tracer=tracer, cap=cap, recon=recon, base_dir=base_dir)
            if space == 'correlation':
                estimate_correlation_function(path_data, path_randoms, path_result, path_shifted=path_shifted, zcuts=environment.zcuts[tracer], nthreads=4, recon=recon)  # downsampling by 2 takes about 5 min
            if space == 'power':
                estimate_power_spectrum(path_data, path_randoms, path_result, path_shifted=path_shifted, zcuts=environment.zcuts[tracer])

    if 'average' in run:
        path_results = [environment.path_measurement(space=space, tracer=tracer, cap=cap, recon=recon, base_dir=base_dir) for cap in environment.caps]
        path_result = environment.path_measurement(space=space, tracer=tracer, recon=recon, base_dir=base_dir)
        path_ref = environment.path_measurement(space=space, tracer=tracer, recon=recon)
        ref = Measurement.load(path_ref)
        if space == 'correlation':
            average_correlation_function(path_results, environment.effective_area[tracer], path_result, cov=ref.cov, attrs=ref.attrs)
        if space == 'power':
            average_power_spectrum(path_results, environment.effective_area[tracer], path_result, cov=ref.cov, attrs=ref.attrs)

    path_result = environment.path_measurement(space=space, tracer=tracer, recon=recon, base_dir=base_dir)
    path_ref = environment.path_measurement(space=space, tracer=tracer, recon=recon)
    linestyles = ['-', ':']
    labels = [None, 'ref']
    path_figure = os.path.join(base_dir,'eBOSS_{}_{}-vDR16.png'.format(tracer, space))
    if 'plot' in run:
        if space == 'correlation':
            #path_ref = environment.path_measurement(space=space, tracer=tracer, recon=False, base_dir=base_dir)
            plot_correlation_function([path_result, path_ref], labels=labels, linestyles=linestyles, path_figure=path_figure)
            #plot_correlation_function(path_result, path_figure=path_figure)
        if space == 'power':
            plot_power_spectrum([path_result, path_ref], labels=labels, linestyles=linestyles, path_figure=path_figure)
