# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:32:32 2020

@author: julbr
"""

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

chemin = 'spoken_digit_dataset/'

def genere_nom(nbr_prononce, locuteur, essai):
    chemin_total = chemin +str(nbr_prononce)+"_"+locuteur+"_"+str(essai)+".wav"
    return chemin_total

samplerate, data = wav.read(genere_nom(1, "jackson", 25))
print(samplerate, data)
length = data.shape[0] / samplerate
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data, label="Channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()