#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:50:42 2020

@author: raphael
"""

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

chemin = "spoken_digit_dataset/"
loc  = ["jackson", "jason", "nicolas", "theo"]

def genere_nom(c, l, o):
    return chemin + str(c) + '_' + loc[l] + '_' + str(o) + '.wav'

if __name__ == '__main__':
    filename = genere_nom(3, 0, 0)
    # Fs : int, Sample rate of wav file
    # s : numpy array, Data read from wav file
    [Fs, s] = wav.read(filename)
    nb_sounds = len(s)
    t = np.linspace(0, nb_sounds/Fs, nb_sounds)
    plt.plot(t, s)