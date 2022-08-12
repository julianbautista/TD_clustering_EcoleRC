# Repo for exercises at Euclid summer school

## Installation
```
pip install matplotlib jupyter cython mpi4py
python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,astropy]
python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]
python -m pip install git+https://github.com/adematti/pmesh
python -m pip install git+https://github.com/cosmodesi/pypower
```
Optionally:
```
pip install emcee
pip install corner
```

Quick tests:
```
git clone https://github.com/adematti/TD_clustering
cd TD_clustering
python tests.py
```

## Y1: Hands on eBOSS LRG (plus BOSS CMASS) sample

See eBOSS_LRGpCMASS.ipynb

## Y2: Cosmological constraints with BAO

See bao_cosmo.ipynb

## Answers

See answers/

### Downloads (eBOSS catalogs and measurements)
```
pip install requests  # to download catalogs
python environment.py
```
