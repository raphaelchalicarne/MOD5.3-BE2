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
import time as time

# %% Paramètres globaux
np.random.seed(13)
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
    y = np.arange(0.5, ordre_modele+1, 1)
    lpc_plot = ax.pcolormesh(x, y, matrice_lpc.T)
    fig.colorbar(lpc_plot, ax=ax)
    return None


def calcul_matrice_distances_lpc(mat_lpc_1, mat_lpc_2, plot=False):
    """
    Parameters
    ----------
    mat_lpc_1 : numpy array
        Matrice de coefficients LPC d'un signal audio (obtenu par `calcul_lpc`).
    mat_lpc_2 : numpy array
        Matrice de coefficients LPC d'un signal audio (obtenu par `calcul_lpc`).

    Returns
    -------
    distance_elastique : float
        Distance cumulée minimale du chemin entre la distance en (0,0) et la distance en (-1,-1).
    """
    nb_elements_1, ordre = mat_lpc_1.shape
    nb_elements_2, _ = mat_lpc_2.shape
    mat_distances = np.zeros((nb_elements_1, nb_elements_2))
    distance_min = np.zeros((nb_elements_1, nb_elements_2))
    for i in range(nb_elements_1):
        for j in range(nb_elements_2):
            mat_distances[i, j] = np.sqrt(
                ((mat_lpc_1[i]-mat_lpc_2[j])**2).sum())
        # Matrice de distance euclidienne entre chaque frame des matrices des
        # coefficients LPC de deux signaux audio. La valeur en i,j correspond
        # à la distance euclidienne entre la frame i de mat_lpc_1 et la frame j de mat_lpc_2.
            if i == 0 or j == 0:
                distance_min[i, j] = mat_distances[i, j]
            else:
                distance_min[i, j] = mat_distances[i, j] + np.min(
                    [distance_min[i-1, j-1], distance_min[i-1, j], distance_min[i, j-1]])
    distance_min = distance_min / \
        (nb_elements_1 + nb_elements_2)  # Normalisation
    distance_elastique = distance_min[-1, -1]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Distance frame à frame entre les deux signaux")
        ax.set_xlabel(r"Frame $j$ du second signal")
        ax.set_ylabel(r"Frame $i$ du premier signal")
        distance_plot = ax.pcolormesh(mat_distances)
        fig.colorbar(distance_plot, ax=ax)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Distance minimale cumulée entre les deux signaux")
        ax.set_xlabel(r"Frame $j$ du second signal")
        ax.set_ylabel(r"Frame $i$ du premier signal")
        distance_min_plot = ax.pcolormesh(distance_min)
        fig.colorbar(distance_min_plot, ax=ax)

    return distance_elastique

# %% Fonctions de test pour calculer la distance entre signaux


def calcul_lpc_01_loc0():
    ordre_modele = 10
    index_locuteur = 0
    matrices_lpc = np.empty((2, 50), dtype=object)
    for nbr_prononce in range(2):
        for essai in range(50):
            filename = genere_nom(nbr_prononce, index_locuteur, essai)
            samplerate, data = wav.read(filename)
            matrice_lpc = calcul_lpc(data, samplerate, ordre_modele)
            matrices_lpc[nbr_prononce, essai] = matrice_lpc
    return matrices_lpc


def calcul_distance_01_loc0(matrices_lpc):
    nb_essai, nb_nbr_prononce = matrices_lpc.shape
    nb_distances = nb_nbr_prononce*nb_essai
    distances_matrices_lpc = np.zeros((nb_distances, nb_distances))
    for i in range(nb_distances):
        for j in range(nb_distances):
            nbr_prononce_i = i % nb_essai
            essai_i = i//nb_essai
            nbr_prononce_j = j % nb_essai
            essai_j = j//nb_essai
            distance_ij = calcul_matrice_distances_lpc(
                matrices_lpc[nbr_prononce_i, essai_i], matrices_lpc[nbr_prononce_j, essai_j])
            distances_matrices_lpc[i, j] = distance_ij
    return distances_matrices_lpc
# %%


def matrices_lpc_locuteur(index_locuteur=0, nbr_essais=5, ordre_modele=10):
    """
    Parameters
    ----------
    index_locuteur : int
        Index du locuteur dont on va mesurer la distance des signaux aux 
        signaux des autres locuteurs.
    nbr_essais : int
        Nombre de signaux sélectionnés par locuteur et par nombre prononcé.

    Returns
    -------
    matrices_lpc : numpy array
        Matrice contenant la matrice des coefficients LPC de chaque signal témoin.
    index_essais : numpy array
        Liste des des essais considérés pour les signaux.

    """
    t0 = time.time()

    range_4 = np.arange(4)
    locuteurs = range_4[range_4 != index_locuteur] #Liste des index des locuteurs "témoin"
    #Liste des index des enregistrements considérés pour chaque locuteur et chaque nombre prononcé
    index_essais = np.random.choice(50, nbr_essais, replace=False)

    matrices_lpc = np.empty((10, 3*nbr_essais), dtype=object)
    for nbr_prononce in range(10):
        for i_locuteur, locuteur in enumerate(locuteurs):
            for i_essai, essai in enumerate(index_essais):
                filename = genere_nom(nbr_prononce, locuteur, essai)
                samplerate, data = wav.read(filename)
                # On peut retrouver à chaque ligne la matrice LPC correspondant à un nombre prononcé
                # Chaque colonne correspond d'abord à un locuteur, puis au numéro de l'enregistrement considéré
                matrices_lpc[nbr_prononce, i_locuteur*nbr_essais +
                             i_essai] = calcul_lpc(data, samplerate, ordre_modele)

    print("Temps de calcul matrices LPC :",
          "{0:.2f}".format(time.time()-t0), "secondes")
    return matrices_lpc, index_essais


def calcul_kppv_locuteur(matrices_lpc, index_essais, index_locuteur=0, ordre_modele=10):
    """
    

    Parameters
    ----------
    matrices_lpc : numpy array
        Matrice contenant la matrice des coefficients LPC de chaque signal témoin.
    index_essais : numpy array
        Liste des des essais considérés pour les signaux.
    index_locuteur : int, optional
        Index du locuteur test, dont on va comparer les enregistrements aux autres locuteurs. 
        La valeur par défaut est 0.
    ordre_modele : Int, optional
        Ordre du modèle de prédiction linéaire LPC. Correspond au nombre de
        frames que l'on souhaite considérer pour prédire le signal. 
        La valeur par défaut est 10.

    Returns
    -------
    classe_kppv : numpy array
        Matrice donnant la classe calculée (nombre que l'on suppose avoir été prononcé)
        avec en ligne le nombre réellement prononcé (valeur témoin)
        et en colonne l'enregistrement considéré.

    """
    t0 = time.time()

    classe_kppv = np.zeros((10, len(index_essais)), dtype="int8")
    # On itère selon les enregistrements du locuteur test.
    for nbr_prononce_test in range(10):
        for i_essai_test, essai_test in enumerate(index_essais):
            filename_test = genere_nom(
                nbr_prononce_test, index_locuteur, essai_test)
            samplerate_test, data_test = wav.read(filename_test)
            # Cette matrice des coefficients LPC sera comparée à toutes les matrices témoins.
            mat_lpc_test = calcul_lpc(data_test, samplerate_test, ordre_modele)

            # Cette matrice contient la distance entre `mat_lpc_test` et la matrice témoin.
            # On retrouve en abcisse le locuteur et le numéro d'enregistrement, et en ordonnée le numéro prononcé.
            distances = np.zeros((10, 3*len(index_essais)))

            # On itère selon les enregistrements des locuteurs témoin.
            for nbr_prononce in range(10):
                for i_essai_temoin in range(3*len(index_essais)):
                    mat_lpc_temoin = matrices_lpc[nbr_prononce, i_essai_temoin]
                    distance_test_temoin = calcul_matrice_distances_lpc(
                        mat_lpc_test, mat_lpc_temoin)
                    distances[nbr_prononce,
                              i_essai_temoin] = distance_test_temoin

            if i_essai_test==0:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title("Distances de la matrice" + str(nbr_prononce_test) +
                             '_' + str(index_locuteur) + '_' + str(essai_test))
                ax.set_xlabel(r"Essai témoin $i$")
                ax.set_ylabel("Nombre prononcé")
                distances_plot = ax.pcolormesh(distances)
                fig.colorbar(distances_plot, ax=ax)

            # 10% des distances sont inférieures à `percentile_10_pourcents`.
            percentile_10_pourcents = np.percentile(distances, 10)
            # Liste des classes où la distance est inférieure au premier décile.
            classes_distances_min = np.where(
                distances < percentile_10_pourcents)[0]
            (values, counts) = np.unique(
                classes_distances_min, return_counts=True)
            index_classe_argmax = np.argmax(counts)
            # Classe pour laquelle le plus de distances sont inférieures au premier décile.
            # On considère qu'il s'agit du nombre prononcé dans le signal test.
            classe = values[index_classe_argmax]

            classe_kppv[nbr_prononce_test, i_essai_test] = classe

    print("Temps de calcul classe_kppv :",
          "{0:.2f}".format(time.time()-t0), "secondes")
    return classe_kppv


if __name__ == '__main__':
    ordre_modele = 10
    ##### Partie 1 #####
    # filename_532 = genere_nom(5, 3, 2)
    # filename_533 = genere_nom(5, 3, 3)
    # samplerate_532, data_532 = wav.read(filename_532)
    # samplerate_533, data_533 = wav.read(filename_533)

    # matrice_lpc_532 = calcul_lpc(data_532, samplerate_532, ordre_modele)
    # matrice_lpc_533 = calcul_lpc(data_533, samplerate_533, ordre_modele)
    # mat_distances_532_533, distance_532_533 = calcul_matrice_distances_lpc(
    #     matrice_lpc_532, matrice_lpc_533, plot=True)

    # plot_lpc(5, 3, 2)
    # plot_lpc(5, 3, 3)

    ##### Partie 2 (test) #####
    # matrices_lpc = calcul_lpc_01_loc0()
    # distances_matrices_lpc = calcul_distance_01_loc0(matrices_lpc)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title("Distances élastiques entre signaux")
    # ax.set_xlabel(r"Signal $j$")
    # ax.set_ylabel(r"Signal $i$")
    # distance_lpc_plot = ax.pcolormesh(distances_matrices_lpc)
    # fig.colorbar(distance_lpc_plot, ax=ax)

    ##### Partie 3 #####
    i_locuteur = 0
    matrices_lpc_loc3, index_essais = matrices_lpc_locuteur(index_locuteur=i_locuteur, nbr_essais=5)
    # np.save('matrices_lpc_loc3', matrices_lpc_loc3) # Save the numpy array to a npy file.
    # np.save('index_essais', index_essais)

    ##### Partie 4 #####
    # matrices_lpc_loc3 = np.load('matrices_lpc_loc3.npy')
    # index_essais = np.load('index_essais.npy')
    classe_kppv_3 = calcul_kppv_locuteur(matrices_lpc_loc3, index_essais, index_locuteur=i_locuteur)
    np.save('classe_kppv_3', classe_kppv_3)
