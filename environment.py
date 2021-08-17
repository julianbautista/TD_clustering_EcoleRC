import os
import requests

import numpy as np

data_dir = './data'
catalog_dir = './catalogs'
catalog_origin_dir = os.path.join(catalog_dir,'origin')

caps = ['NGC','SGC']
effective_area = {}
effective_area['LRG'] = [2476., 1627.] # for NGC, SGC
effective_area['LRGpCMASS'] = [272226., 129825.] # actually number of objects
effective_redshift = {}
effective_redshift['LRGpCMASS'] = 0.698
zcuts = {'LRGpCMASS':(0.6,1.0)}


class Measurement(object):

    _attrs = ['x','y','cov','attrs']

    def __init__(self, x, y, cov=None, attrs=None):
        self.attrs = attrs or {}
        self.x = x
        self.y = y
        self.cov = cov

    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    copy = __copy__

    def __getstate__(self):
        return {name:getattr(self,name,None) for name in self._attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        self = cls.__new__(cls)
        self.__setstate__(state)
        return self

    def save(self, path):
        np.save(path,self.__getstate__())

    @classmethod
    def load(cls, path):
        self = object.__new__(cls)
        self.__setstate__(np.load(path,allow_pickle=True)[()])
        return self


def mkdir(path):
	try: os.makedirs(path) #MPI...
	except OSError: return


def load_ref_data(path_data):
    tmp = np.loadtxt(path_data,unpack=True)
    return tmp[0],tmp[1::2]


def load_ref_covariance(path_covariance):
    cols = ['recon1','recon2','ell1','ell2','x1','x2','m']
    converters = {}
    converters['recon1'] = lambda x: x.strip()
    converters['recon2'] = converters['recon1']
    converters['ell1'] = lambda x: int(x)
    converters['ell2'] = converters['ell1']
    converters['x1'] = lambda x: float(x)
    converters['x2'] = converters['x1']
    converters['m'] = converters['x1']
    file = open(path_covariance,'r')
    toret = {c:[] for c in cols}
    for line in file:
        if line.startswith('#'): continue
        line = line.split()
        line = [word for word in line if word]
        norecon = len(line) < 7
        if norecon:
            for word,col in zip(line,['ell1','ell2','x1','x2','m']):
                toret[col].append(converters[col](word))
            toret['recon1'].append(0)
            toret['recon2'].append(0)
        else:
            for word,col in zip(line,cols):
                toret[col].append(converters[col](word))

    for col in cols: toret[col] = np.array(toret[col])

    def to_square(toret):
        nx = int(round(len(toret['x1'])**0.5))
        toretm = np.ones((nx,nx),dtype='f8')
        conv,xcov = {},{}
        nx = 0

        def get_uniques(tab):
            toret = []
            for t in tab:
                if t not in toret:
                    toret.append(t)
            return toret

        urecon = get_uniques(toret['recon1'])
        uells = get_uniques(toret['ell1'])

        for recon in urecon:
            conv[recon] = {}
            xcov[recon] = {}
            for ell in uells:
                conv[recon][ell] = {}
                mask = (toret['recon1'] == recon) & (toret['ell1'] == ell)
                xcov[recon][ell] = get_uniques(toret['x1'][mask])
                for x in xcov[recon][ell]:
                    conv[recon][ell][x] = nx
                    nx += 1
        for im,m in enumerate(toret['m']):
            ibin1 = conv[toret['recon1'][im]][toret['ell1'][im]][toret['x1'][im]]
            ibin2 = conv[toret['recon2'][im]][toret['ell2'][im]][toret['x2'][im]]
            toretm[ibin1][ibin2] = toret['m'][im]
        new = {}
        new['m'] = toretm
        new['recons'] = urecon
        new['ells'] = uells
        new['xs'] = xcov[recon][ell]
        return new

    return to_square(toret)



def download_measurement(space='power', tracer='LRGpCMASS', recon=False, base_dir=data_dir):

    label_recon = 'postrecon' if recon else 'prerecon'
    if tracer == 'LRGpCMASS':
        if space == 'correlation':
            url = 'https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/dataveccov/lrg_elg_qso/LRG_xi/'
            base_data = ['Data_LRGxi_NGCSGC_0.6z1.0_{}.txt'.format(label_recon)]
            base_covariance = ['Covariance_LRGxi_NGCSGC_0.6z1.0_{}.txt'.format(label_recon)]
        else:
            url = 'https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/dataveccov/lrg_elg_qso/LRG_Pk/'
            base_data = ['Data_LRGPk_{}_0.6z1.0_{}.txt'.format(cap,label_recon) for cap in ['NGC','SGC']]
            base_covariance = ['Covariance_LRGPk_{}_0.6z1.0_{}.txt'.format(cap,label_recon) for cap in ['NGC','SGC']]

    path_ref_data = [os.path.join(base_dir,bd) for bd in base_data]
    path_ref_covariance = [os.path.join(base_dir,bd) for bd in base_covariance]

    mkdir(base_dir)

    for bd,prd in zip(base_data + base_covariance,path_ref_data + path_ref_covariance):
        urld = os.path.join(url,bd)
        r = requests.get(urld,allow_redirects=True)
        with open(prd,'bw') as file: file.write(r.content)
    
    coeffs = effective_area[tracer]
    coeffs = np.array(coeffs)/np.sum(coeffs)

    toret = 0
    sum_coeffs = 0
    for path_ref,coeff in zip(path_ref_data,coeffs):
        x,multi = load_ref_data(path_ref)
        toret += coeff*np.array([x] + list(multi))
        sum_coeffs += coeff
    toret /= sum_coeffs
    coeffs /= sum_coeffs
    x, y = toret[0], toret[1:]

    cov = 0
    for path_ref,coeff in zip(path_ref_covariance,coeffs):
        mat = load_ref_covariance(path_ref)['m']
        cov += coeff**2*mat

    zeff = effective_redshift[tracer]

    ells = [0,2] if recon else [0,2,4]
    m = Measurement(x,y,cov=cov,attrs={'space':space,'tracer':tracer,'zeff':zeff,'recon':recon,'ells':ells})
    m.save(path_measurement(space=space,tracer=tracer,recon=recon,base_dir=base_dir))


def path_measurement(space='power', tracer='LRGpCMASS', recon=False, base_dir=data_dir):
    if recon: recon = '_rec'
    else: recon = ''
    base = '{}_eBOSS_{}{}.npy'.format(space,tracer,recon)
    return os.path.join(base_dir,base)


def download_catalogs(tracer='LRGpCMASS', cap='NGC', recon=False, base_dir=catalog_dir):
    url_dir = 'https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/'
    url_data, url_randoms = path_catalogs(tracer=tracer,cap=cap,recon=recon,base_dir=url_dir)
    base_data, base_randoms = path_catalogs(tracer=tracer,cap=cap,recon=recon,base_dir=base_dir)

    mkdir(base_dir)
    for urld,fn in zip([url_data,url_randoms],[base_data, base_randoms]):
        r = requests.get(urld,allow_redirects=True)
        with open(fn,'bw') as file: file.write(r.content)


def path_catalogs(tracer='LRGpCMASS', cap='NGC', recon=False, base_dir=catalog_dir):
    if recon: recon = '_rec'
    else: recon = ''
    base_data = 'eBOSS_{}_clustering_data{}-{}-vDR16.fits'.format(tracer,recon,cap)
    base_randoms = 'eBOSS_{}_clustering_random{}-{}-vDR16.fits'.format(tracer,recon,cap)
    return os.path.join(base_dir,base_data), os.path.join(base_dir,base_randoms)


def prune_catalogs(tracer='LRGpCMASS', cap='NGC', recon=False):
    path_origin_data, path_origin_randoms = path_catalogs(tracer=tracer,cap=cap,recon=recon,base_dir=catalog_origin_dir)
    path_data, path_randoms = path_catalogs(tracer=tracer,cap=cap,recon=recon,base_dir=catalog_dir)
    from astropy.table import Table
    catalog = Table.read(path_origin_data)
    catalog.remove_columns(['LRG_ID','SECTOR','WEIGHT_FKP','WEIGHT_FKP_EBOSS','WEIGHT_ALL_NOFKP','IN_EBOSS_FOOT','ISCMASS','WEIGHT_FKP_CMASS'])
    print(catalog.columns)
    print('Data size is {:d}'.format(len(catalog)))
    catalog.write(path_data,overwrite=True)
    catalog = Table.read(path_origin_randoms)
    catalog.remove_columns(['SECTOR','WEIGHT_FKP','WEIGHT_FKP_EBOSS','WEIGHT_ALL_NOFKP','IN_EBOSS_FOOT','ISCMASS','WEIGHT_FKP_CMASS'])
    print(catalog.columns)
    rng = np.random.RandomState(seed=42)
    index = rng.uniform(0.,1.,len(catalog)) < 1./5.
    catalog = catalog[index]
    print('Randoms size is {:d}'.format(len(catalog)))
    catalog.write(path_randoms,overwrite=True)



if __name__ == '__main__':

    tracer = 'LRGpCMASS'
    for recon in [False,True]:
        #for cap in ['NGC','SGC']:
        #    download_catalogs(tracer=tracer,cap=cap,recon=recon,base_dir=catalog_origin_dir)
        #for cap in ['NGC','SGC']:
        #    prune_catalogs(tracer=tracer,cap=cap,recon=recon)
        for space in ['power','correlation']:
            download_measurement(space=space,tracer=tracer,recon=recon)
