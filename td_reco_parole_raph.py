#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:50:42 2020

@author: raphael
"""
#%% Imports
import scipy.io.wavfile as wav
from scipy.linalg import toeplitz
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#%% Paramètres globaux
chemin = "spoken_digit_dataset/"
locuteurs  = ["jackson", "jason", "nicolas", "theo"]

#%% Fonctions
def genere_nom(nbr_prononce, index_locuteur, essai):
    return chemin + str(nbr_prononce) + '_' + locuteurs[index_locuteur] + '_' + str(essai) + '.wav'

def calcul_lpc_fenetre(fenetre, ordre):
    """
    Parameters
    ----------
    fenetre : Array
        DESCRIPTION.
    ordre : Int
        DESCRIPTION.

    Returns
    -------
    TYPE
        Coefficients lpc pour une fenêtre donnée.

    """
    t_fenetre = len(fenetre)
    R = np.zeros((ordre + 1, 1))
    for k in range(ordre + 1):
        R[k] = np.mean(fenetre[0:t_fenetre - k - 1] * fenetre[k:t_fenetre - 1])
    m_R = toeplitz(R)
    
    v = np.zeros((ordre + 1, 1))
    v[0] = 1
    
    lpc = np.dot(inv(m_R), v)
    lpc = lpc/lpc[0]
    return lpc[1:]

def calcul_lpc(data, samplerate, ordre, taille_fenetre=0.02):
    taille_data_fenetre = int(samplerate*taille_fenetre)
    taille_decalage = taille_data_fenetre//2
    taille_data = data.shape[0]
    taille_mat_lpc = taille_data//taille_decalage
    matrice_lpc = np.zeros((taille_mat_lpc, ordre))
    for k in range(taille_mat_lpc):
        fenetre_inf = k*taille_decalage
        fenetre_sup = fenetre_inf + taille_data_fenetre
        lpc = calcul_lpc_fenetre(data[fenetre_inf:fenetre_sup], ordre)
        lpc = lpc.ravel()
        matrice_lpc[k] = lpc
    return matrice_lpc

def calcul_matrice_distances_lpc(mat_lpc_1, mat_lpc_2):
    nb_elements_1, ordre = mat_lpc_1.shape
    nb_elements_2, _ = mat_lpc_2.shape
    mat_distances = np.zeros((nb_elements_1, nb_elements_2))
    for i in range(nb_elements_1):
        for j in range(nb_elements_2):
            mat_distances[i,j] = np.sqrt(((mat_lpc_1[i]-mat_lpc_2[j])**2).sum())
    return mat_distances

if __name__ == '__main__':
    ordre_modele = 10
    # filename = genere_nom(3, 0, 0)
    filename_532 = genere_nom(5,3,2)
    filename_533 = genere_nom(5,3,3)
    samplerate_532, data_532 = wav.read(filename_532)
    samplerate_533, data_533 = wav.read(filename_533)

    # length = data.shape[0] / samplerate
    # time = np.linspace(0., length, data.shape[0])
    # plt.plot(time, data, label=filename)
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.show()
    
    matrice_lpc_532 = calcul_lpc(data_532, samplerate_532, ordre_modele)
    matrice_lpc_533 = calcul_lpc(data_533, samplerate_533, ordre_modele)
    distances_532_533 = calcul_matrice_distances_lpc(matrice_lpc_532, matrice_lpc_533)

    
    
    