#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:47:20 2020

@author: dellandrea
"""

import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import inv

chemin = 'spoken_digit_dataset/'
loc = ['jackson', 'jason', 'nicolas', 'theo']

def genere_nom(c,l,o):
    return chemin + str(c) + '_' + loc[l] + '_' + str(o) + '.wav'

def calcul_lpc_fenetre(f,ordre_modele):
    
    t_fenetre = len(f)
    R = np.zeros((ordre_modele+1,1))
    for k in range(ordre_modele+1):
        R[k] = np.mean(f[0:t_fenetre-1-k] * f[k:t_fenetre-1])
        
    m_R = toeplitz(R)
    
    v = np.zeros((ordre_modele+1,1))
    v[0] = 1
    
    lpc = np.dot(inv(m_R),v)
    lpc = lpc / lpc[0]

    return lpc[1:]

def calcul_lpc(s,Fe,ordre_modele):
    
    t_signal = len(s)
    t_fenetre = int(Fe/50)
    
    offset = 0
    f = s[offset:offset+t_fenetre]       
    coeff =calcul_lpc_fenetre(f,ordre_modele)
    offset += t_fenetre //2    

    while offset + t_fenetre <= t_signal:
        f = s[offset:offset+t_fenetre]       
        coeff = np.hstack((coeff,calcul_lpc_fenetre(f,ordre_modele)))
        
        offset += t_fenetre // 2
        
    return coeff

def distance_elastique(c1,c2):
    pass
    

if __name__ == '__main__':
    
    ordre_modele = 10
    Fe, s1 = wav.read(genere_nom(1,1,2))
    Fe, s2 = wav.read(genere_nom(5,3,2))

#    print(s.shape)
#    plt.plot(s)
#    plt.show()
    
#    lpc = calcul_lpc_fenetre(s[0:160],ordre_modele)
    coeff1 = calcul_lpc(s1,Fe,ordre_modele)
    coeff2 = calcul_lpc(s2,Fe,ordre_modele)
    
    print(coeff1.shape)
    print(coeff2.shape)
    
    dist = distance_elastique(coeff1,coeff2)
    