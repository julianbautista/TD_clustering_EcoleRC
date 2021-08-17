import os
import numpy as np

from cosmoprimo import Cosmology

import likelihood_bao
from likelihood_bao import PowerBAOLikelihood, minimize
import filepaths
from filepaths import Measurement


base_dir = './sims'
zeff = 0.1 + np.linspace(0.,1.5,4)


def get_cosmo_sim():
    cosmo_kwargs = dict(Omega_m=0.26,omega_b=0.022,h=0.72,sigma8=0.8,n_s=0.97,N_ur=2.0328,m_ncdm=[0.06])
    #cosmo_kwargs = dict(Omega_m=0.31,omega_b=0.022,h=0.676,sigma8=0.8,n_s=0.97,N_ur=2.0328,m_ncdm=[0.06])
    cosmo = Cosmology(**cosmo_kwargs,engine='class')
    return cosmo


def generate_sim_model(likelihood, x, base_measurement, path_measurement, zeff=None, seed=None):

    rng = np.random.RandomState(seed=seed)
    cosmo = get_cosmo_sim()
    ba_fid = likelihood.cosmo.get_background()
    zeff0 = likelihood.zeff
    if zeff is None: zeff = zeff0
    if np.ndim(zeff) == 0:
        zeff = [zeff]
        path_measurement = [path_measurement]

    for z,path in zip(zeff,path_measurement):
        likelihood.zeff = z
        likelihood.set_model(cosmo=cosmo)
        ba = cosmo.get_background()
        amp = ba.growth_factor(z)/ba.growth_factor(zeff0)

        hubble_rate_fid = ba_fid.efunc(z)
        comoving_angular_distance_fid = ba_fid.comoving_angular_distance(z)
        hubble_rate = ba.efunc(z)
        comoving_angular_distance = ba.comoving_angular_distance(z)
        qpar = hubble_rate_fid/hubble_rate
        qperp = comoving_angular_distance/comoving_angular_distance_fid
        d = likelihood.list_to_dict(x)
        if likelihood.iso:
            d['qiso'] = qpar**(1./3.)*qperp**(2./3.)
        else:
            d['qpar'] = qpar
            d['qperp'] = qperp
        x = likelihood.dict_to_list(**d)
        y = likelihood.model(x,k=base_measurement.x)/amp**2
        cov = base_measurement.cov[:y.size,:y.size]*0.1
        y_ = rng.multivariate_normal(y.flatten(),cov)
        #y_ = y
        measurement = base_measurement.copy()
        measurement.y = y_.reshape(y.shape)
        measurement.cov = cov
        measurement.attrs['zeff'] = z
        measurement.save(path)


def path_sim_measurement(space='power', tracer='LRGpCMASS', z=0.5, recon=False, base_dir=base_dir):
    if recon: recon = '_rec'
    else: recon = ''
    base = '{}_sim_{}{}_{:.2f}.npy'.format(space,tracer,recon,z)
    return os.path.join(base_dir,base)


def list_path_sim_measurement(**kwargs):
    return [path_sim_measurement(z=z,**kwargs) for z in zeff]


if __name__ == '__main__':

    filepaths.mkdir(base_dir)

    tracer = 'LRGpCMASS'
    iso = True

    path_data = filepaths.path_measurement(space='power',tracer=tracer)
    likelihood = PowerBAOLikelihood(path_data,iso=iso)

    x = minimize(likelihood)

    base_measurement = Measurement.load(path_data)
    paths = list_path_sim_measurement(space='power',tracer=tracer)
    generate_sim_model(likelihood,x,base_measurement,paths,zeff=zeff,seed=42)
