# EEG-sleep-analysis

Tests of different approaches to EEG sleep analysis, on individual polysomnograpic recordings from the publicly available [Sleep-EDF Expanded](https://physionet.org/pn4/sleep-edfx/) database. The recordings and annotations can be downloaded from the [Physionet ATM](https://physionet.org/cgi-bin/atm/ATM) and should be present in the /data folder.
Currently only contains classification based on the discrete Hidden Markov model (DHMM) and feature extraction in the frequency domain (different band ratios).

Command to run the HMM:

$ python3.6 src/dhmm.py data/SC4001E0-PSG.edf data/annotations.txt

