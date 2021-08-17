import os
import logging

import numpy as np
from scipy import special
import Corrfunc


class PairCount(object):

    def __init__(self, wnpairs, total_wnpairs=1.):
        self.wnpairs = wnpairs
        self.total_wnpairs = total_wnpairs

    def set_total_wnpairs(self, w1, w2=None):
        if w2 is not None:
            self.total_wnpairs = np.sum(w1)*np.sum(w2)
        else:
            self.total_wnpairs = np.sum(w1)**2 - np.sum(w1**2)

    def __getstate__(self):
        return {'wnpairs':self.wnpairs,'total_wnpairs':self.total_wnpairs}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def normalized(self):
        return self.wnpairs/self.total_wnpairs

    @property
    def shape(self):
        return self.wnpairs.shape

    @classmethod
    def from_state(cls, state):
        self = cls.__new__(cls)
        self.__setstate__(state)
        return self


class CorrelationFunction(object):

    logger = logging.getLogger('CorrelationFunction')

    def __init__(self, data, randoms, data2=None, randoms2=None, sedges=None, ells=(0,2,4), nmu=100, nthreads=1, distance=None, R1R2=None, attrs=None):
        self.attrs = attrs or {}
        self.data1 = data
        self.randoms1 = randoms
        self.data2 = data2 if data2 is not None else self.data1
        self.randoms2 = randoms2 if randoms2 is not None else self.randoms1
        self.sedges = np.asarray(sedges)
        self.muedges = np.linspace(0.,1.,nmu+1)
        self.ells = ells
        self.nthreads = nthreads
        if distance is not None:
            for cat in [self.data1,self.data2,self.randoms1,self.randoms2]:
                cat['DISTANCE'] = distance(cat['Z'])
        self.R1R2 = R1R2
        if isinstance(R1R2,str):
            self.load_RR(path_R1R2=R1R2)
        self.run()

    def load_RR(self, path_R1R2=None):
        self.R1R2 = None
        if path_R1R2 is not None and os.path.isfile(path_R1R2):
            try:
                self.R1R2 = self.__class__.load(path_R1R2).R1R2
                self.logger.info('Loading {}: {}.'.format(self.__class__.__name__,path_R1R2))
            except:
                self.R1R2 = np.load(self.path_R1R2)
                self.logger.info('Loading {}: {}.'.format(self.R1R2.__class__.__name__,path_R1R2))
        else:
            self.logger.info('File R1R2 {} not found. It will be recomputed.'.format(path_R1R2))

    def run(self):
        names = ['D1D2','D1R2']
        cats = [(self.data1,self.data2),(self.data1,self.randoms2)]
        if getattr(self,'R1R2',None) is None:
            names.append('R1R2')
            cats.append((self.randoms1,self.randoms2))
        if self.data2 is not self.data1 or self.randoms2 is not self.randoms1:
            names.append('D2R1')
            cats.append((self.data2,self.randoms1))
        for name,(cat1,cat2) in zip(names,cats):
            autocorr = cat2 is cat1
            self.logger.info('Computing {} pair counts.'.format(name))
            result = Corrfunc.mocks.DDsmu_mocks(autocorr,cosmology=1,nthreads=self.nthreads,mu_max=1.,nmu_bins=len(self.muedges)-1,binfile=self.sedges,
                                                RA1=cat1['RA'],DEC1=cat1['DEC'],CZ1=cat1['DISTANCE'],weights1=cat1['WEIGHT'],
                                                RA2=cat2['RA'],DEC2=cat2['DEC'],CZ2=cat2['DISTANCE'],weights2=cat2['WEIGHT'],
                                                is_comoving_dist=True,verbose=True,output_savg=False,weight_type='pair_product')
            pc = PairCount(result['npairs']*result['weightavg'])
            pc.wnpairs.shape = (len(self.sedges)-1,len(self.muedges)-1)
            pc.set_total_wnpairs(cat1['WEIGHT'],w2=None if cat2 is cat1 else cat2['WEIGHT'])
            setattr(self,name,pc)
        if 'D2R1' not in names:
            self.D2R1 = self.D1R2
        self.set_ximu()
        self.set_xiell()

    def set_ximu(self):
        self.set_sep()
        nonzero = self.R1R2.wnpairs > 0
        # init
        self.xismu = np.zeros(self.D1D2.shape)
        self.xismu[:] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized()[nonzero]
        DR = self.D1R2.normalized()[nonzero]
        RD = self.D2R1.normalized()[nonzero]
        RR = self.R1R2.normalized()[nonzero]
        corr = (DD - DR - RD)/RR + 1
        self.xismu[nonzero] = corr[:]

    def set_xiell(self):
        self.xiell = []
        dmu = np.diff(self.muedges)
        for ell in self.ells:
            legendre = (2*ell + 1) * special.legendre(ell)(self.muedges)
            legendre = (legendre[:-1] + legendre[1:])/2.
            self.xiell.append(np.sum(self.xismu*legendre*dmu,axis=-1)/np.sum(dmu))
        self.xiell = np.array(self.xiell)

    def set_sep(self):
        self.s = (self.sedges[1:] + self.sedges[:-1])/2.
        self.mu = (self.muedges[1:] + self.muedges[:-1])/2.

    def __call__(self, sep, ell=0):
        return np.interp(sep,self.sep,self.xiell[self.ells.index(ell)],left=0.,right=0.)

    def __getstate__(self):
        state = {}
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if hasattr(self,key): state[key] = getattr(self,key).__getstate__()
        for key in ['attrs','edges','s','mu','xismu','sedges','muedges','ells']:
            if hasattr(self,key): state[key] = getattr(self,key)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if hasattr(self,key): state[key] = PairCount.from_state(getattr(self,key))
        self.set_xiell()

    def save(self, path):
        self.logger.info('Saving to: {}.'.format(path))
        np.save(path,self.__getstate__())

    @classmethod
    def load(cls,path):
        cls.logger.info('Loading: {}.'.format(path))
        self = object.__new__(cls)
        self.__setstate__(np.load(path,allow_pickle=True)[()])
        return self
