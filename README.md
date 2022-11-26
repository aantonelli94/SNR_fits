# Predicting the SNR of IMBHs with next-gen GW interferometers.


This repo contains codes to predict the signal-to-noise ratio (SNR) of intermediate-mass black holes (IMBHs) using a configuration of next-generation ground-based detectors. In particular, a configuration of two Cosmic Explorer (of 20km and 40km) and the Einstein Telescope is assumed.

## What the repo contains

The repo contains a python notebook `analysis.ipynb` with an analysis of the polynomial regression model chosen for this task, and a library `xgsnr.py` that can be used as specified in the usage below.

## Usage

To predict the SNR, the total mass $M$, symmetric mass ratio $0 <\eta < 0.25$ and redshift $z$ of the binary must be input, as follows


```
# Simple usage to predict the fit given masses and redshift.

from xgsnr import *

Mtot = 1e3 # Total mass in solar masses
eta = 0.2 # Symmetric mass ratio in solar masses (between 0 and 0.25)
z = 1

SNR_fit(Mtot,eta,z)
```


## Requirements

To run the repo on your computer, you must have the following libraries installed.

```
numpy==1.21.5
pandas==1.4.3
scikit_learn==1.1.3
```

## Report a bug

Please contact andrico@hotmail.it if you would like to report a bug, or if you have any feedback or comments.
