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

def distance_element_a_element(matrice_lpc_1, matrice_lpc_2):
    """
    on considere la matrice 1 en colonne et matrice 2 en ligne
    on calcule la distance euclidienne entre chaque vecteur colonne des deux matrices
    """
    taille1, taille2 = np.shape(matrice_lpc_1)[0], np.shape(matrice_lpc_2)[0]
    matrice_distance = np.zeros((taille1, taille2))
    for i in range(taille1):
        elt1 = matrice_lpc_1[i,:]
        for j in range(taille2):
            elt2 = matrice_lpc_2[j,:]
            distance = np.sqrt(np.sum((elt1-elt2)**2))
            matrice_distance[i,j] = distance
    return matrice_distance

def distance_elastique(matrice_lpc_1, matrice_lpc_2):
    cout_vert = 1 #correspond a w_v
    cout_horiz = 1 #correspond a w_h
    cout_diag = 1 #correspond a w_d
    matrice_distance = distance_element_a_element(matrice_lpc_1, matrice_lpc_2)
    taille1, taille2 = np.shape(matrice_distance)
    matrice_distance_elastique = np.zeros((taille1, taille2))
    
    #initialisation premier element
    matrice_distance_elastique[0,0] = matrice_distance[0,0]
    #initialisation colonne 1
    for i in range(1,taille1):
        matrice_distance_elastique[i,0] = matrice_distance_elastique[i-1,0] + cout_vert*matrice_distance[i,0]
    #initialisation ligne 1
    for j in range(1,taille2):
        matrice_distance_elastique[0,j] = matrice_distance_elastique[0,j-1] + cout_horiz*matrice_distance[0,j]

    # remplissage du reste
    for i in range(1,taille1):
        for j in range(1,taille2):
            chemin_vert = matrice_distance_elastique[i-1,j] + cout_vert*matrice_distance[i,j]
            chemin_horiz = matrice_distance_elastique[i,j-1] + cout_horiz*matrice_distance[i,j]
            chemin_diag = matrice_distance_elastique[i-1,j-1] + cout_diag*matrice_distance[i,j]
            matrice_distance_elastique[i,j] = min(chemin_vert, chemin_horiz, chemin_diag)
    return matrice_distance_elastique[-1,-1] /(taille1+taille2)

#%% test
nbr_prononce = 5
locuteur=liste_nom[0]
essai = 2

N = 10
chemin1 = genere_nom(nbr_prononce, locuteur, essai)
chemin2 = genere_nom(5, liste_nom[0], 2)
samplerate1, data1 = wav.read(chemin1)
samplerate2, data2 = wav.read(chemin2)
if data1.ndim > 1 :
    data1 = data1[:,0]
    
#print(np.shape(calcul_coeff_lpc_fenetre(data[0:160], 10)))
matrice1 =calcul_lpc(samplerate1, data1, 10)
matrice2 =calcul_lpc(samplerate2, data2, 10)
truc_a_print = distance_elastique(matrice1, matrice2)
print(truc_a_print)
#affichage(chemin)