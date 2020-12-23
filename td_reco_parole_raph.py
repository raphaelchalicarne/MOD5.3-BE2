#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:50:42 2020

@author: raphael
"""
# %% Imports
import scipy.io.wavfile as wav
from scipy.linalg import toeplitz
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# %% Paramètres globaux
chemin = "spoken_digit_dataset/"
locuteurs = ["jackson", "jason", "nicolas", "theo"]

# %% Fonctions


def genere_nom(nbr_prononce, index_locuteur, essai):
    """
    Parameters
    ----------
    nbr_prononce : Int
        Nombre prononcé sur l'enregistrement.
    index_locuteur : Int
        Index du prénom du locuteur dans la liste `locuteurs`.
    essai : Int
        Numéro de l'enregistrement du fichier audio.

    Returns
    -------
    String
        Chemin d'accès à un fichier audio utilisé pour la prédiction des 
        coefficients LPC.

    """
    return chemin + str(nbr_prononce) + '_' + locuteurs[index_locuteur] + '_' + str(essai) + '.wav'


def plot_signal(nbr_prononce, index_locuteur, essai):
    filename = genere_nom(nbr_prononce, index_locuteur, essai)
    samplerate, data = wav.read(filename)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data, label=filename)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    return None


def calcul_lpc_fenetre(fenetre, ordre):
    """
    Parameters
    ----------
    fenetre : Array
        Signal sur une fenêtre temporelle de l'ordre de 20 ms.
    ordre : Int
        Ordre du modèle de prédiction linéaire LPC. Correspond au nombre de
        frames que l'on souhaite considérer pour prédire le signal.

    Returns
    -------
    TYPE
        Coefficients lpc a_1 à a_ordre pour une fenêtre donnée.

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
    """
    Parameters
    ----------
    data : numpy array
        Data read from wav file. Data-type is determined from the file.
    samplerate : int
        Sample rate of wav file.
    ordre : int
        Ordre du modèle de prédiction linéaire LPC. Correspond au nombre de
        frames que l'on souhaite considérer pour prédire le signal.
    taille_fenetre : float, optional
        Taille en secondes d'une fenêtre temporelle. La valeur par défaut 0.02.

    Returns
    -------
    matrice_lpc : numpy array
        Matrice dont chaque ligne correspond à une frame, et chaque colonne à un coefficient LPC.
    """
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


def plot_lpc(nbr_prononce, index_locuteur, essai, ordre_modele=10):
    """
    Affiche avec Matplotlib les coefficients LPC d'un modèle d'ordre 
    `ordre_modele` calculés sur un enregistrement audio, frame par frame.

    Parameters
    ----------
    nbr_prononce : Int
        Nombre prononcé sur l'enregistrement.
    index_locuteur : Int
        Index du prénom du locuteur dans la liste `locuteurs`.
    essai : Int
        Numéro de l'enregistrement du fichier audio.
    ordre_modele : Int, optional
        Ordre du modèle de prédiction linéaire LPC. Correspond au nombre de
        frames que l'on souhaite considérer pour prédire le signal. 
        La valeur par défaut est 10.

    Returns
    -------
    None.

    """
    filename = genere_nom(nbr_prononce, index_locuteur, essai)
    samplerate, data = wav.read(filename)
    matrice_lpc = calcul_lpc(data, samplerate, ordre_modele)
    taille_mat_lpc = matrice_lpc.shape[0]

    fig, ax = plt.subplots()
    ax.set_title("Coefficients LPC du fichier " + str(nbr_prononce) +
                 '_' + locuteurs[index_locuteur] + '_' + str(essai) + '.wav')
    ax.set_xlabel("Frame")
    ax.set_ylabel(r"Coefficient $a_{i}$")
    x = np.arange(-0.5, taille_mat_lpc, 1)
    y = np.arange(0.5, ordre_modele+1, 1)  # len = 11
    lpc_plot = ax.pcolormesh(x, y, matrice_lpc.T)
    fig.colorbar(lpc_plot, ax=ax)
    return None


def calcul_matrice_distances_lpc(mat_lpc_1, mat_lpc_2):
    """
    Parameters
    ----------
    mat_lpc_1 : numpy array
        Matrice de coefficients LPC d'un signal audio (obtenu par `calcul_lpc`).
    mat_lpc_2 : numpy array
        Matrice de coefficients LPC d'un signal audio (obtenu par `calcul_lpc`).

    Returns
    -------
    mat_distances : numpy array
        Matrice de distance euclidienne entre chaque frame des matrices des coefficients LPC de deux signaux audio.
        La valeur en i,j correspond à la distance euclidienne entre la frame i de mat_lpc_1 et la frame j de mat_lpc_2.
    """
    nb_elements_1, ordre = mat_lpc_1.shape
    nb_elements_2, _ = mat_lpc_2.shape
    mat_distances = np.zeros((nb_elements_1, nb_elements_2))
    for i in range(nb_elements_1):
        for j in range(nb_elements_2):
            mat_distances[i, j] = np.sqrt(
                ((mat_lpc_1[i]-mat_lpc_2[j])**2).sum())
    return mat_distances


if __name__ == '__main__':
    ordre_modele = 10
    # filename = genere_nom(3, 0, 0)
    filename_532 = genere_nom(5, 3, 2)
    filename_533 = genere_nom(5, 3, 3)
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
    distances_532_533 = calcul_matrice_distances_lpc(
        matrice_lpc_532, matrice_lpc_533)
