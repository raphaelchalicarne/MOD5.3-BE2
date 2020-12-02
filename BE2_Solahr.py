# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:32:32 2020

@author: julbr
"""

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz, inv

chemin = 'spoken_digit_dataset/'
liste_nom = ['jackson','jason', 'nicolas', 'theo']

def genere_nom(nbr_prononce, locuteur, essai):
    chemin_total = chemin +str(nbr_prononce)+"_"+locuteur+"_"+str(essai)+".wav"
    return chemin_total

def affichage(chemin):
#    chemin = genere_nom(nbr_prononce, locuteur, essai)
    samplerate, data = wav.read(chemin)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data, label=chemin)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    
#affichage(1, "theo", 47)
    
def calcul_matrice_R(liste, N): # liste est  un np.array        N ordre du moodele
    liste_R =[np.mean(liste*liste)]
    for i in range(1,N+1):
        liste1 = liste[:-i]
        liste2 = liste[i:]
        liste_R.append(np.mean(liste1*liste2))
    return toeplitz(liste_R)

def calcul_coeff_lpc_fenetre(data, N):
    matrice_R = calcul_matrice_R(data, N)
    vecteur_sigma = np.zeros((N+1,1))
    vecteur_sigma[0,0] = 1
    matrice_R_inv = inv(matrice_R)
    coeff_LPC= matrice_R_inv.dot(vecteur_sigma)
    coeff_LPC/=coeff_LPC[0,0]
    return coeff_LPC[1:]
    
def calcul_lpc(samplerate, data, ordre_modele, taille_fenetre=0.02): #taille fenetre en seconde
    nombre_element_fenetre = int(taille_fenetre*samplerate)
    decalage_fenetre = nombre_element_fenetre//2
    longueur_echantillon = len(data)
    coeff_lpc_matrix = []
#    coeff_lpc_matrix = np.zeros((ordre_modele, longueur_echantillon//decalage_fenetre))
    for i in range(0,longueur_echantillon,decalage_fenetre):
        data_fenetre = data[i:i+nombre_element_fenetre]
#        coeff_lpc_matrix[:, i] = calcul_coeff_lpc_fenetre(data_fenetre, ordre_modele)[:,0]
#        coeff_lpc_matrix+=calcul_coeff_lpc_fenetre(data_fenetre, ordre_modele)
        coeff_lpc_matrix.append(calcul_coeff_lpc_fenetre(data_fenetre, ordre_modele)[:,0])
    return np.array(coeff_lpc_matrix)

#%% test
nbr_prononce = 5
locuteur=liste_nom[3]
essai = 2

N = 10
chemin = genere_nom(nbr_prononce, locuteur, essai)
samplerate, data = wav.read(chemin)
#print(np.shape(calcul_coeff_lpc_fenetre(data[0:160], 10)))
#print(np.shape(calcul_lpc(samplerate, data, 10)))
affichage(chemin)