import os
import logging

import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from nbodykit import setup_logging
from nbodykit.transform import SkyToCartesian
from nbodykit.lab import *

from correlation_function import CorrelationFunction
import environment
from environment import Measurement


logger = logging.getLogger('Analysis')

base_dir = './results'


def get_cosmo_BOSS():
    cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,sigma8=0.8,n_s=0.97,N_ur=2.0328,m_ncdm=[0.06])
    cosmo_kwargs['Omega0_b'] = cosmo_kwargs.pop('omega_b')/cosmo_kwargs['h']**2
    Omega0_m = cosmo_kwargs.pop('Omega_m')
    sigma8 = cosmo_kwargs.pop('sigma8')
    cosmo = cosmology.Cosmology(**cosmo_kwargs).match(Omega0_m=Omega0_m).match(sigma8=sigma8)
    return cosmo


def estimate_correlation_function(path_data, path_randoms, path_result, nthreads=4, zcuts=None, downsample_randoms=10, recon=False):
    data = Table.read(path_data)
    randoms = Table.read(path_randoms)
    P0 = 1e4
    for catalog in [data,randoms]:
        catalog['WEIGHT_FKP'] = 1./(1. + catalog['NZ']*P0)
        catalog['WEIGHT'] = catalog['WEIGHT_SYSTOT']*catalog['WEIGHT_CP']*catalog['WEIGHT_NOZ']*catalog['WEIGHT_FKP']
    if zcuts is not None:
        data = data[(data['Z'] > zcuts[0]) & (data['Z'] < zcuts[1])]
        randoms = randoms[(randoms['Z'] > zcuts[0]) & (randoms['Z'] < zcuts[1])]
    rng = np.random.RandomState(seed=42)
    index = rng.uniform(0.,1.,len(randoms)) < 1./downsample_randoms
    randoms = randoms[index]
    sedges = np.linspace(1e-9,140,40)
    distance = get_cosmo_BOSS().comoving_distance
    correlation = CorrelationFunction(data=data,randoms=randoms,sedges=sedges,distance=distance,ells=(0,2,4),nthreads=nthreads)
    if recon: # no recon should be computed first
        correlation.R1R2 = CorrelationFunction.load(path_result.replace('_rec','')).R1R2
    correlation.save(path_result)


def plot_correlation_function(path_results, labels=None, linestyles='-', path_figure=None):
    if isinstance(path_results,str):
        path_results = [path_results]
        labels = [labels]
        linestyles = [linestyles]

    ax = plt.gca()
    for path_result,label,linestyle in zip(path_results,labels,linestyles):
        data = Measurement.load(path_result)
        for ill,ell in enumerate(data.attrs['ells']):
            label_ = '$\ell = {:d}$'.format(ell)
            if ill == 0 and label is not None:
                label_ = '{} {}'.format(label,label_)
            ax.plot(data.x,data.x**2*data.y[ill],label=label_,linestyle=linestyle,color='C{:d}'.format(ill))

    ax.legend(loc=1,fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    ax.set_xlabel('$s$ [$\\mathrm{Mpc} / h$]',fontsize=17)
    ax.set_ylabel('$s^{2} \\xi(s)$ [$(\\mathrm{{Mpc}} / h)^{2}$]',fontsize=17)
    if path_figure is not None:
        logger.info('Saving figure to {}.'.format(path_figure))
        plt.savefig(path_figure,bbox_inches='tight',pad_inches=0.1,dpi=200)


def estimate_power_spectrum(path_data, path_randoms, path_result, zcuts=None, downsample_randoms=1):
    data = FITSCatalog(path_data)
    randoms = FITSCatalog(path_randoms)
    cosmo = get_cosmo_BOSS()
    P0 = 1e4
    for catalog in [data,randoms]:
        catalog['WEIGHT_FKP'] = 1./(1. + catalog['NZ']*P0)
        catalog['WEIGHT_COMP'] = catalog['WEIGHT_SYSTOT']*catalog['WEIGHT_CP']*catalog['WEIGHT_NOZ']
        catalog['POSITION'] = SkyToCartesian(catalog['RA'],catalog['DEC'],catalog['Z'],cosmo=cosmo)
    if zcuts is not None:
        data = data[(data['Z'] > zcuts[0]) & (data['Z'] < zcuts[1])]
        randoms = randoms[(randoms['Z'] > zcuts[0]) & (randoms['Z'] < zcuts[1])]
    from nbodykit.mpirng import MPIRandomState
    rng = MPIRandomState(comm=randoms.comm,seed=42,size=randoms.size)
    index = rng.uniform(0.,1.) < 1./downsample_randoms
    #randoms = randoms[index]
    fkp = FKPCatalog(data,randoms)
    BoxSize = 3000.
    Nmesh = 100
    mesh = fkp.to_mesh(position='POSITION',fkp_weight='WEIGHT_FKP',comp_weight='WEIGHT_COMP',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
    power = ConvolvedFFTPower(mesh,poles=(0,2,4),kmin=0.,dk=0.01)
    power.save(path_result)


def plot_power_spectrum(path_results, labels=None, linestyles='-', path_figure=None):

    if isinstance(path_results,str):
        path_results = [path_results]
        labels = [labels]
        linestyles = [linestyles]

    ax = plt.gca()
    for path_result,label,linestyle in zip(path_results,labels,linestyles):
        data = Measurement.load(path_result)
        for ill,ell in enumerate(data.attrs['ells']):
            label_ = '$\ell = {:d}$'.format(ell)
            if ill == 0 and label is not None:
                label_ = '{} {}'.format(label,label_)
            ax.plot(data.x,data.x*data.y[ill],label=label_,linestyle=linestyle,color='C{:d}'.format(ill))

    ax.legend(loc=1,fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    ax.set_xlabel('$k$ [$h / \\mathrm{Mpc}$]',fontsize=17)
    ax.set_ylabel('$k P(k)$ [$(\\mathrm{{Mpc}} / h)^{2}$]',fontsize=17)
    if path_figure is not None:
        logger.info('Saving figure to {}.'.format(path_figure))
        plt.savefig(path_figure,bbox_inches='tight',pad_inches=0.1,dpi=200)


def average_correlation_function(path_results, effective_area, path_average, cov=None, attrs=None):
    attrs = attrs or {}
    y = 0
    for path_result,area in zip(path_results,effective_area):
        correlation = CorrelationFunction.load(path_result)
        ells = correlation.ells
        x = correlation.s
        y += correlation.xiell*area
    y /= sum(effective_area)
    m = Measurement(x,y,cov=cov,attrs={**attrs,**{'space':'correlation','ells':ells}})
    m.save(path_average)


def average_power_spectrum(path_results, effective_area, path_average, cov=None, attrs=None):
    attrs = attrs or {}
    y = 0
    for path_result,area in zip(path_results,effective_area):
        power = ConvolvedFFTPower.load(path_result)
        poles = power.poles
        ells = power.attrs['poles']
        shotnoise = power.attrs['shotnoise']
        x = poles['k']
        y_ = []
        for ill,ell in enumerate(ells):
            pk = poles['power_{:d}'.format(ell)].real
            if ell == 0: pk -= shotnoise
            y_.append(pk)
        y += np.array(y_)*area
    y /= sum(effective_area)
    m = Measurement(x,y,cov=cov,attrs={**attrs,**{'space':'power','ells':ells}})
    m.save(path_average)


def path_measurement(space='power', tracer='LRGpCMASS', cap=None, recon=False, base_dir=base_dir):
    if recon: recon = '_rec'
    else: recon = ''
    if cap is None: cap = ''
    else: cap = '_{}'.format(cap)
    base = '{}_eBOSS_{}{}{}.npy'.format(space,tracer,cap,recon)
    return os.path.join(base_dir,base)



if __name__ == '__main__':

    setup_logging()
    environment.mkdir(base_dir)
    tracer = 'LRGpCMASS'
    run = ['estimator','average','plot'][2:]
    space = 'correlation'
    #space = 'power'
    recon = False
    #recon = True

    if 'estimator' in run:
        path_results = []
        for cap in environment.caps:
            path_data, path_randoms = environment.path_catalogs(tracer=tracer,cap=cap,recon=recon)
            path_result = path_measurement(space=space,tracer=tracer,cap=cap,recon=recon)
            if space == 'correlation':
                estimate_correlation_function(path_data,path_randoms,path_result,zcuts=environment.zcuts[tracer],nthreads=4,downsample_randoms=2,recon=recon) # downsampling by 2 takes about 5 min
            if space == 'power':
                path_result = path_result.replace('.npy','.json')
                estimate_power_spectrum(path_data,path_randoms,path_result,zcuts=environment.zcuts[tracer],downsample_randoms=1)

    if 'average' in run:
        path_results = [path_measurement(space=space,tracer=tracer,cap=cap,recon=recon) for cap in environment.caps]
        path_result = path_measurement(space=space,tracer=tracer,recon=recon)
        path_ref = environment.path_measurement(space=space,tracer=tracer,recon=recon)
        ref = Measurement.load(path_ref)
        if space == 'correlation':
            average_correlation_function(path_results,environment.effective_area[tracer],path_result,cov=ref.cov,attrs=ref.attrs)
        if space == 'power':
            path_results = [path_result.replace('.npy','.json') for path_result in path_results]
            average_power_spectrum(path_results,environment.effective_area[tracer],path_result,cov=ref.cov,attrs=ref.attrs)

    path_result = path_measurement(space=space,tracer=tracer,recon=recon,base_dir=base_dir)
    path_ref = environment.path_measurement(space=space,tracer=tracer,recon=recon)
    linestyles = ['-',':']
    labels = [None,'ref']
    path_figure = os.path.join(base_dir,'eBOSS_{}_{}-vDR16.png'.format(tracer,space))
    if 'plot' in run:
        if space == 'correlation':
            #plot_correlation_function([path_result,path_ref],labels=labels,linestyles=linestyles,path_figure=path_figure)
            plot_correlation_function(path_result,path_figure=path_figure)
        if space == 'power':
            plot_power_spectrum([path_result,path_ref],labels=labels,linestyles=linestyles,path_figure=path_figure)
