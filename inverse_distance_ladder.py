import os

import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo import Cosmology

import likelihood_bao
from likelihood_bao import BaseLikelihood, get_cosmo_BOSS, minimize, sample, plot_samples
import filepaths
from filepaths import Measurement

base_dir = './results'


class BAOLikelihood(BaseLikelihood):

    def __init__(self, path_data):
        if isinstance(path_data,str):
            path_data = [path_data]
        self.set_data_covariance(path_data)
        self.set_model()

    def set_data_covariance(self, path_data):
        self.data,self.cov,self.invcov,self.iso,self.zeff = [],[],[],[],[]
        for path in path_data:
            data = Measurement.load(path)
            self.data.append(data.y)
            self.cov.append(data.cov)
            self.invcov.append(np.linalg.inv(data.cov))
            self.iso.append(data.attrs['iso'])
            self.zeff.append(data.x)
        self.zeff = np.array(self.zeff)

    def set_model(self):
        self.params = {}
        self.params['varied'] = ['Omega_m','H0','rs_drag']
        self.params['bounds'] = [(0.1,0.5),(50,100),(100,200)]
        self.params['optional'] = {}
        self.cosmo_fid = get_cosmo_BOSS()
        ba = self.cosmo_fid.get_background()
        th = self.cosmo_fid.get_thermodynamics()
        self.hubble_rate_fid = ba.efunc(self.zeff)
        self.comoving_angular_distance_fid = ba.comoving_angular_distance(self.zeff)
        self.rs_drag_fid = th.rs_drag

    def model(self, args, z=None, iso=None):
        if z is None:
            z = self.zeff
            iso = self.iso
            hubble_rate_fid = self.hubble_rate_fid
            comoving_angular_distance_fid = self.comoving_angular_distance_fid
        else:
            ba = self.cosmo_fid.get_background()
            hubble_rate_fid = ba.efunc(z)
            comoving_angular_distance_fid = ba.comoving_angular_distance(z)
        params = self.list_to_dict(args)
        rs_drag = params.pop('rs_drag')
        cosmo = Cosmology(**params,engine='class')
        rs_drag *= cosmo['h']
        ba = cosmo.get_background()
        qpar = hubble_rate_fid*self.rs_drag_fid/ba.efunc(z)/rs_drag
        qperp = ba.comoving_angular_distance(z)*self.rs_drag_fid/comoving_angular_distance_fid/rs_drag
        toret = []
        for iso_,qpar_,qperp_ in zip(iso,qpar,qperp):
            if iso_: toret.append(qpar_**(1./3.)*qperp_**(2./3.))
            else: toret.append((qpar_,qperp_))
        return toret

    def chi2(self, args):
        model = self.model(args)
        toret = 0.
        for d,m,icov in zip(self.data,model,self.invcov):
            diff = d - m
            toret += diff.dot(icov).T.dot(diff)
        return toret

    def logprior(self, args):
        logprior = super(BAOLikelihood,self).logprior(args)
        d = self.list_to_dict(args)
        # Planck2018 TT lowE
        logprior -= 1./2.*(d['rs_drag'] - 147.21)**2/0.48**2
        return logprior

    def plot(self, params, path_figure=None):
        z = np.linspace(1e-9,2,100)
        qparqperp = np.array(self.model(params,z=z,iso=[False]*len(z))).T
        qiso = np.array(self.model(params,z=z,iso=[True]*len(z))).T
        model = np.concatenate([qiso[None,:],qparqperp],axis=0)
        ax = plt.gca()
        for iq,label in enumerate(['parallel','transverse','iso']):
            ax.plot(z,model[iq],color='C{:d}'.format(iq),label=label)
        for iz,z in enumerate(self.zeff):
            iso, d, c = self.iso[iz], self.data[iz], self.cov[iz]
            if iso:
                ax.errorbar(z,d,c[0]**0.5,linestyle='none',color='C{:d}'.format(iq))
            else:
                ax.errorbar([z]*2,d,np.diag(c)**0.5,linestyle='none',color='C{:d}'.format(iq))
        ax.legend(loc=1,fontsize=15)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.set_xlabel('$z$',fontsize=17)
        ax.set_ylabel(r'$D(z)/r_{\mathrm{drag}}/\left(D^{\mathrm{fid}}(z)/r_{\mathrm{drag}}^{\mathrm{fid}}\right)$',fontsize=17)

        if path_figure is not None:
            plt.savefig(path_figure,bbox_inches='tight',pad_inches=0.1,dpi=200)
            plt.close(plt.gcf())


if __name__ == '__main__':

    filepaths.mkdir(base_dir)

    tracer = 'LRGpCMASS'
    iso = True
    run = ['minimize','mcmc']
    #data = 'eBOSS'
    data = 'sims'

    if data == 'eBOSS':
        path_data = likelihood_bao.path_measurement(space='power',tracer=tracer,iso=iso)
        path_figure = os.path.join(base_dir,'fit_{}_inverse_distance_ladder.png'.format(data))
        path_samples = os.path.join(base_dir,'samples_{}_inverse_distance_ladder.npy'.format(data))
        path_figure_samples = os.path.join(base_dir,'samples_{}_inverse_distance_ladder.png'.format(data))

    if data == 'sims':
        path_data = likelihood_bao.list_path_sim_measurement(space='power',tracer=tracer,iso=iso)
        path_figure = os.path.join(base_dir,'fit_{}_inverse_distance_ladder.png'.format(data))
        path_samples = os.path.join(base_dir,'samples_{}_inverse_distance_ladder.npy'.format(data))
        path_figure_samples = os.path.join(base_dir,'samples_{}_inverse_distance_ladder.png'.format(data))

    likelihood = BAOLikelihood(path_data)

    if 'minimize' in run:
        x = minimize(likelihood)
        msg = '{} are {}'.format(likelihood.varied,x)
        print(msg)
        likelihood.plot(x,path_figure=path_figure)

    if 'mcmc' in run:
        samples = sample(likelihood,nsteps=2000,path_samples=path_samples)
        samples = np.load(path_samples)
        samples = samples[len(samples)//2:]
        x = np.median(samples,axis=0)
        cov = np.cov(samples.T,ddof=1)
        likelihood.plot(x,path_figure=path_figure)
        plot_samples(likelihood,samples,path_figure_samples)
