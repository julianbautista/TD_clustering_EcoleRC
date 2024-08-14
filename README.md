# Repo for exercises given at the Euclid summer school

Everything (including installation of packages and data) should be included in the notebooks.

## Y1: Hands on eBOSS LRG (plus BOSS CMASS) sample

See eBOSS_LRGpCMASS.ipynb.

### Google Colab

Go to https://colab.research.google.com/.
Click on "import" (then "GitHub") and paste https://github.com/adematti/TD_clustering/blob/main/eBOSS_LRGpCMASS.ipynb, then click on the lens on the right.
The notebook will open, click "Copy to Drive" (to be able to make changes).

On Google Colab, the package installation with pip is slow. You may want to download the prepared environment (first cell).
Else, execute the first cell (package installation) as soon as possible (takes some time).

## Y2: Cosmological constraints with BAO

See cosmo_bao.ipynb.

### Google Colab

Go to https://colab.research.google.com/.
Click on "import" (then "GitHub") and paste https://github.com/adematti/TD_clustering/blob/main/cosmo_bao.ipynb, then click on the lens on the right.
The notebook will open, click "Copy to Drive" (to be able to make changes).

## Answers

See answers/

## Installation
```
python -m pip install matplotlib cython mpi4py fitsio
python -m pip install git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,astropy]
USE_GPU=0 python -m pip install git+https://github.com/cosmodesi/pycorr#egg=pycorr[corrfunc]
python -m pip install git+https://github.com/cosmodesi/pypower#egg=pypower[extras]
```
Optionally:
```
python -m pip install emcee corner
```

Quick tests:
```
git clone https://github.com/adematti/TD_clustering
cd TD_clustering
python tests.py
```

### Downloads (eBOSS catalogs and measurements)

#### Lite versions
```
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1lN0xu7mWuu46POSaJ1t5tKCmtxOETfXh?usp=sharing
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1egpLxKnteOQgYIetNDk7TFmk-vDx_O11?usp=sharing
```

#### Full versions
```
pip install requests  # to download catalogs
python environment.py
```
