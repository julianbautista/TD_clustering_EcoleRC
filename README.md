# eBOSS LRG clustering

Collection of routines to treat eBOSS DR16 LRG data.

## Data

Catalogs: https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/

## Instructions

1) Download catalogs

2) Plot footprint : RA / DEC et Z distribution

3) RA, DEC, Z to cartesian positions assuming a fiducial cosmology
Compute FKP weights wtigh P0 = 10000 [Mpc/h]^3.

4) Estimate correlation function of the pre-reconstruction catalogs:
DD, DR, RR pair counts as a function of (s, mu) using corrfunc
How to normalize them?
Build Landy-Szalay estimator.
Estimate multipoles.

5) Repeat 4) with post-reconstruction catalogs.
Compare pre- and post-reconstruction.

6) Estimate power spectrum using nbodykit.
Be careful about memory usage!
All together: recall the algorithm: assignment scheme, interlacing, FFTs, compensation for assignment scheme, normalization, shot noise

7) Code up isotropic BAO model (Fourier space or configuration space).
Fit isotropic BAO parameter alpha.
For sampling you can use e.g. https://github.com/minaskar/zeus

8) Publish!

## Codes

### To estimate the correlation function: corrfunc

https://corrfunc.readthedocs.io/en/master/api/Corrfunc.mocks.html#Corrfunc.mocks.DDsmu_mocks

### To estimate the power spectrum: nbodykit

https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.algorithms.convpower.fkp.html
https://nbodykit.readthedocs.io/en/0.1.11/algorithms/survey-power.html

## References

arXiv:2007.08994
