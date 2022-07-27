import logging

import numpy as np

from cosmoprimo.fiducial import BOSS
from pycorr import TwoPointCorrelationFunction, setup_logging
from pypower import CatalogFFTPower


logger = logging.getLogger('Tests')


def generate_catalogs(size=10000, boxsize=(1000,)*3, offset=(1000.,0.,0.), n_individual_weights=1, seed=42):
    rng = np.random.RandomState(seed=seed)
    positions = [o + rng.uniform(0., 1., size)*b for o, b in zip(offset, boxsize)]
    weights = [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
    return positions, weights


def test_cosmoprimo():
    cosmo = BOSS(engine='class')
    cosmo.comoving_radial_distance(1.0)


def test_pycorr():
    data_positions1, data_weights1 = generate_catalogs(size=10000, seed=42)
    randoms_positions1, randoms_weights1 = generate_catalogs(size=50000, seed=43)
    edges = (np.linspace(0., 100., 51), np.linspace(-1., 1., 201))
    result = TwoPointCorrelationFunction('smu', edges, data_positions1=data_positions1, data_weights1=data_weights1,
                                         randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                                         engine='corrfunc', nthreads=2)

def test_pypower():
    data_positions1, data_weights1 = generate_catalogs(size=10000, seed=42)
    randoms_positions1, randoms_weights1 = generate_catalogs(size=50000, seed=43)
    result = CatalogFFTPower(data_positions1=data_positions1, data_weights1=data_weights1,
                             randoms_positions1=randoms_positions1, randoms_weights1=randoms_weights1,
                             edges=np.linspace(0, 0.2, 11), ells=(0, 2, 4), interlacing=2, boxsize=2000., nmesh=128, resampler='tsc',
                             los=None, position_type='xyz', mpiroot=0)


if __name__ == '__main__':
    # To activate logging
    setup_logging()

    test_pycorr()
    test_pypower()
    logger.info('All good!')
