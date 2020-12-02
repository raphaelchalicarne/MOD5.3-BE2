#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:03:04 2019
@author: Raphaël Chalicarne
@Adapted from https://github.com/YannickJadoul/Parselmouth
"""
# %% Importations

import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly 
import plotly.graph_objs as go

# %% Fonctions

# Fonctions de démonstration du module Parselmouth adaptées de
# https://parselmouth.readthedocs.io/en/stable/examples/plotting.html

def path_show_ext(fullpath):
    """
    splits a full file path into path, basename and extension
    :param fullpath: str
    :return: the path, the basename and the extension
    """
    tmp = os.path.splitext(fullpath)
    ext = tmp[1]
    p = tmp[0]
    while tmp[1] != '':
        tmp = os.path.splitext(p)
        ext = tmp[1] + ext
        p = tmp[0]

    path = os.path.dirname(p)
    if path == '':
        path = '.'
    base = os.path.basename(p)
    return path, base, ext

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")
    
def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")
    
# Analyse du pitch en dessinant l'histogramme des valeurs de pitch sur la droite

def draw_pitch_histogram(snd, title=None):
    """Plot the pitch contour of a Parselmouth sound with the spectrogram
    in the background, and the histogram of the pitch values on the right"""
    
    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    
    # definitions for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.1, 0.65
    spacing = 0.105
    
    rect_specto = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    plt.figure()
    
    ax_specto = plt.axes(rect_specto)
    ax_specto.tick_params(direction='in', right=True)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    # draw_spectrogram(spectrogram)
    dynamic_range = 70
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    ax_specto.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    ax_specto.set_xlim([snd.xmin, snd.xmax])
    ax_specto.set_ylim([spectrogram.ymin, spectrogram.ymax])
    ax_specto.set_xlabel("time [s]")
    ax_specto.set_ylabel("frequency [Hz]")
    
    ax_pitch = ax_specto.twinx()
    
    # draw_pitch(pitch)
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values_max = pitch_values.max()
    pitch_values[pitch_values==0] = np.nan
    ax_pitch.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    ax_pitch.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    ax_pitch.grid(False)
    ax_pitch.set_ylim(0, pitch.ceiling)
    ax_pitch.set_ylabel("fundamental frequency [Hz]")
    
    # Histogram
    # now determine nice limits by hand:
    binwidth = 5
    lim = np.ceil(pitch_values_max / binwidth) * binwidth
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy.hist(pitch_values, bins=bins, orientation='horizontal', facecolor= u'b', edgecolor = u'b')
    ax_histy.set_ylim(ax_pitch.get_ylim())
    
    plt.title(title)
    
    plt.show()
    
def pitch_clusterisation(snd):
    """Input : a parselmouth sound
    Output : Liste des indices de début de pitch
    Extraits de pitch"""
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    indices_debut = []
    p = []
    j = -1
    pitch_values[-1]=0
    for i, p_value in enumerate(pitch_values):
        if p_value != 0 and i > j:
            i_debut = i
            j = i_debut
            indices_debut.append(i_debut)
            while pitch_values[j] != 0:
                j+=1
            p.append(pitch_values[i_debut:j])
    return indices_debut, p

def pitch_contour_analysis(pitch, binwidth = 5):
    """Input : array de float représentant les fréquences fondamentales d'un pitch
    Output :    Fréquence minimale du pitch,
                Fréquence maximale du pitch
                Fréquences revenant le plus souvent"""
    bin_min = np.floor(pitch.min() / binwidth) * binwidth
    bin_max = np.ceil(pitch.max() / binwidth) * binwidth
    bins = np.arange(bin_min, bin_max + binwidth, binwidth)
    
    hist = np.histogram(pitch, bins)[0]
    indices_freq_principal = np.where(hist == np.amax(hist))
    bin_principal = bins[indices_freq_principal]
    
    return bin_min, bin_max, bin_principal

def plot_pitch_carac(sound, binwidth = 5):
    indices_debut, p = pitch_clusterisation(sound)
    pitch_caracteristics = []
    data_width = []
    data_max = []
    
    for i, pitch_extract in enumerate(p):
        bin_min, bin_max, bin_principal = pitch_contour_analysis(pitch_extract, binwidth)
        pitch_caracteristics.append([bin_min, bin_max, bin_principal])
        
        points_x = np.array([indices_debut[i],
                    indices_debut[i],
                    indices_debut[i] + len(pitch_extract),
                    indices_debut[i] + len(pitch_extract)])/100
        points_lim = [bin_min, bin_max, bin_max, bin_min]
        
        for p_max in bin_principal:
            points_max = [p_max, p_max + binwidth, p_max + binwidth, p_max]
            data_max+=[go.Scatter(x = points_x,
                          y = points_max,
                          fill='toself',
                          fillcolor = 'red',
                          mode = 'none')]
        
        data_width+=[go.Scatter(x = points_x,
                          y = points_lim,
                          fill='toself',
                          fillcolor = 'blue',
                          mode = 'none')]
        
    layout_speakers = go.Layout(title='Séparation des locuteurs basé sur les pitchs, extrait ' + filename, 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Freqs (Hz)',)) 
    plotly.offline.plot(go.Figure(data = data_width + data_max,
                                layout=layout_speakers),
                                filename= "pitch_separation_" + filename+".html",
                                auto_open=True)
    
def speaker_caracs(sound, binwidth = 5):
    """Input : A parselmouth sound
    Output : Lists of the lower, higher and most regular frequencies of a speaker's 
             pitch contour"""
    indices_debut, p = pitch_clusterisation(sound)
    liste_fmin = []
    liste_fmax = []
    liste_f0 = []
    for i, pitch_extract in enumerate(p):
        bin_min, bin_max, bin_principal = pitch_contour_analysis(pitch_extract, binwidth)
        duree = len(pitch_extract)
        freq_principale = bin_principal[0]
        for j in range(duree):
            liste_fmin.append(bin_min)
            liste_fmax.append(bin_max)
            liste_f0.append(freq_principale)
            
    liste_fmin = np.array(liste_fmin)
    liste_fmax = np.array(liste_fmax)
    liste_f0 = np.array(liste_f0)
    
    return liste_fmin, liste_fmax, liste_f0

def plot_histogramme(liste_frequences, title, binwidth = 5):
    bin_liste_min = np.floor(liste_frequences.min() / binwidth) * binwidth
    bin_liste_max = np.ceil(liste_frequences.max() / binwidth) * binwidth
    bins = np.arange(bin_liste_min, bin_liste_max + binwidth, binwidth)
    
    plt.figure()
    plt.hist(liste_frequences, bins)
    plt.title(title)
    plt.show()
    
def caracs_reco_pitch(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    """Input : Audio recordings filenames list of a same speaker.
    Output : Lists of the lower, higher and most regular frequencies of a speaker's 
             pitch contour"""
    liste_fmin_tot = []
    liste_fmax_tot = []
    liste_f0_tot = []
    for filename in filenames:
        filepath = os.path.join(data_dir, filename + '.wav')
        snd = parselmouth.Sound(filepath)
        liste_fmin, liste_fmax, liste_f0 = speaker_caracs(snd)
        liste_fmin_tot = np.concatenate((liste_fmin_tot, liste_fmin))
        liste_fmax_tot = np.concatenate((liste_fmax_tot, liste_fmax))
        liste_f0_tot = np.concatenate((liste_f0_tot, liste_f0))
    return liste_fmin_tot, liste_fmax_tot, liste_f0_tot

def entrainement_reco_pitch(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    """Entrée : Noms de fichiers audio d'un même locuteur
    Sortie : f_min, f_max et f0 du pitch de ce locuteur"""
    liste_fmin_tot, liste_fmax_tot, liste_f0_tot = caracs_reco_pitch(filenames, data_dir)
        
    fmin_speaker = np.median(liste_fmin_tot)
    fmax_speaker = np.median(liste_fmax_tot)
    f0_speaker = np.median(liste_f0_tot)
    return [fmin_speaker, fmax_speaker, f0_speaker, filenames[0]]

def diarization(filename_test, *f_caracs):
    """Input : Un nom de fichier audio de test qu'on va souhaiter diariser
    *f_caracs : Les caractéristiques (fmin_speaker, fmax_speaker, f0_speaker)
                pour chaque locuteur
    Output : Histogramme de qui parle quand"""
    filepath = os.path.join(data_dir, filename_test + '.wav')
    snd = parselmouth.Sound(filepath)
    
#    weights = np.array([0.25, 0.25, 0.5])
    weights = np.array([0.15, 0.15, 0.7])
    
    pitch_caracteristics = []
    data_width = []
    data_2nd = []
    
    # Séparation de l'audio en différents extraits dont on récupère les pitchs.
    indices_debut, p = pitch_clusterisation(snd)
    scores = np.zeros((len(p),4),float)
    speaker_ids = []
    for i, p_extract in enumerate(p):
        p_min, p_max, p_principal = pitch_contour_analysis(p_extract)
        p_freq = np.array([p_min, p_max, p_principal[0]])
        scores_p = []
        for f_carac_speaker in f_caracs:
            f_carac_speaker_values = np.array(f_carac_speaker[:3])
            score = np.sum(weights*(f_carac_speaker_values - p_freq)**2)
            scores_p.append(score)
        # Le speaker est celui qui a le score le plus bas
        # (frequences plus proches de celles du pitch)
        scores[i] = scores_p
        id_speaker = np.argmin(scores_p)
        name_speaker = f_caracs[id_speaker][-1]
        speaker_ids.append(id_speaker)
        
        
        
        # Plot the result
        pitch_caracteristics.append(p_freq)
        
        points_x = np.array([indices_debut[i],
                    indices_debut[i],
                    indices_debut[i] + len(p_extract),
                    indices_debut[i] + len(p_extract)])/100
        points_y = [id_speaker - 1/2, id_speaker + 1/2, id_speaker + 1/2, id_speaker - 1/2]
        
        if scores_p[id_speaker] < 2000:
            data_width+=[go.Scatter(x = points_x,
                              y = points_y,
                              fill='toself',
                              fillcolor = 'blue',
                              mode = 'none')]
        else:
            data_width+=[go.Scatter(x = points_x,
                              y = points_y,
                              fill='toself',
                              fillcolor = 'blue',
                              opacity = 0.5,
                              mode = 'none')]
    
        # On analyse le deuxième plus petit score, et on l'affiche sur le graphe 
        # s'il est similaire au plus petit score
        id_2nd_speaker = np.argsort(scores_p)[1]
#        if scores_p[id_2nd_speaker] < 2*scores_p[id_speaker]:
        if scores_p[id_2nd_speaker] < scores_p[id_speaker] + 1000:
            points_y2 = [id_2nd_speaker - 1/2, id_2nd_speaker + 1/2, id_2nd_speaker + 1/2, id_2nd_speaker - 1/2]
        
            data_2nd+=[go.Scatter(x = points_x,
                              y = points_y2,
                              fill='toself',
                              fillcolor = 'red',
                              opacity = 0.5,
                              mode = 'none')]
    
    layout_speakers = go.Layout(title='Séparation des locuteurs basé sur les caractéristiques des pitchs, extrait audio ' + filename_test, 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Yuko - MJPM - MJFG - MAB',)) 
    
    plotly.offline.plot(go.Figure(data = data_width + data_2nd,
                                layout=layout_speakers),
                                filename= "pitch_separation_" + filename_test+".html",
                                auto_open=True)
        
    return speaker_ids, scores

# Analyse pitch complets locuteurs
def get_pitch_from_speaker(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    pitch_values = []
    for filename in filenames:
        filepath = os.path.join(data_dir, filename + '.wav')
        snd = parselmouth.Sound(filepath)
        pitch = snd.to_pitch()
        pitch_values_file = pitch.selected_array['frequency']
        pitch_values = np.concatenate((pitch_values,pitch_values_file))
        pitch_values_non_zero = pitch_values[pitch_values > 0]
    return pitch_values_non_zero

def pitch_values_to_bins(pitch_values, binwidth = 5, plot=False, title=None):
    bin_min = 0
    bin_max = 600
    bins = np.arange(bin_min, bin_max, binwidth)
    hist, bin_edges = np.histogram(pitch_values, bins, density=True)
    if plot:
        plt.figure()
        plt.title(title)
        plt.xlabel("Fréquences (Hz)")
        plt.hist(bins[:-1], bins, weights=hist)
    return hist, bin_edges

def filenames_to_pitch_hist(*filenames_list,  data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1'), plot=False):
    hist_pitch_speakers = []
    for filenames in filenames_list:
        pitch_values_speaker = get_pitch_from_speaker(filenames, data_dir)
        hist_pitch_speaker, bins = pitch_values_to_bins(pitch_values_speaker, plot=plot)
        hist_pitch_speakers.append(hist_pitch_speaker)
    tableau_pitchs_speakers = np.vstack(tuple(hist_pitch_speakers))
    return tableau_pitchs_speakers, bins

# Fonctions ACP
def pitch_moyen_extraits_parles(*filenames_lists, data_dir=os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    """Input : Audio recordings filenames list of a same speaker.
    Output : Tableau avec pour variables :  fréquence de pitch minimum
                                            fréquence de pitc maximum
                                            fréquence de pitch le plus récurrent
                                individus : les extraits parlés pondérés par leur longueur
                                
    tableau_donnees est composé de la concaténation verticale des tableaux contenus dans tableaux_locuteurs"""
    # Example of filenames_lists : filenames_MJPM = ['MJPM-1','MJPM-2','MJPM-3']
    tableaux_locuteurs = []
    for filename_list in filenames_lists:
        liste_fmin_tot, liste_fmax_tot, liste_f0_tot = caracs_reco_pitch(filename_list)
        tableau_locuteur = np.vstack((liste_fmin_tot, liste_fmax_tot, liste_f0_tot)).T
        tableaux_locuteurs.append(tableau_locuteur)
        
    tableau_donnees = np.vstack(tuple(tableaux_locuteurs))
    return tableau_donnees, tableaux_locuteurs

def centrer_reduire_tableau(tableau_acp):
    """Input : Tableau d'individus, et pour variables les coefficients MFCC
    Output : Le tableau centré réduit"""
    moyennes_variables = np.mean(tableau_acp, axis=0)
    variances_variables = np.var(tableau_acp, axis=0)
    tableau_centre = tableau_acp - moyennes_variables
    tableau_centre_reduit =  np.divide(tableau_centre, variances_variables, out=np.zeros_like(tableau_centre), where=variances_variables!=0)
    return tableau_centre_reduit

def ACP(tableau_donnees):
    """Input : tableau avec les individus en lignes et les variables quantitatives en colonne
    Output : Valeurs propres et vecteurs propres correspondant aux axes principaux du nuage des individus"""
    tableau_centre_reduit = centrer_reduire_tableau(tableau_donnees)
    # Dimensions du tableau
    # I = nombre d'individus
    # K = nombre de variables
    (I, K) = np.shape(tableau_centre_reduit)
    D = (1/I)*np.eye(I)
    tXDX = tableau_centre_reduit.T.dot(D.dot(tableau_centre_reduit))
    eigenvalues, eigenvectors = np.linalg.eig(tXDX)
    return eigenvalues, eigenvectors

def plot_nuage_individus(eigenvectors, tableaux_locuteurs, speaker_labels):
    colors = ['b', 'r', 'g', 'black', 'c', 'm', 'y', 'k', 'w']
    
    plt.figure()
    for i, tableau_locuteur in enumerate(tableaux_locuteurs):
        coor_F1_locuteur = tableau_locuteur.dot(eigenvectors[0])
        coor_F2_locuteur = tableau_locuteur.dot(eigenvectors[1])
        plt.scatter(coor_F1_locuteur, coor_F2_locuteur, color = colors[i], alpha=0.5, label=speaker_labels[i])
    plt.title("Projection du nuage des individus sur F1 et F2")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()

## Fonctions pour l'API et l'identification des silences / temps parlés

def isSpeakingInThisExtract(pitch_extract, ratio=0.5):
    """Input : pitch_extract is a unidimensionnal numpy array
    Output : A boolean calculated over the proportion of non null values in the extract"""
    len_extract = len(pitch_extract)
    nb_non_zero = np.count_nonzero(pitch_extract)
    return nb_non_zero/len_extract > ratio

def speaking_identification(snd):
    """Input : a parselmouth sound
    Output : Liste de couples [i_debut, i_fin] des moments parlés"""
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']

    len_pitch = len(pitch_values)
    len_output = 2**14

    i_fin = -1
    moments_parles = []
    
    
    if len_pitch <= len_output:
        k = len_output/len_pitch # Rapport > 1
        
        # On s'assure que la boucle while termine.
        pitch_values[-1] = 0
        
        # Dans ce cas on parcourt la liste des valeurs de pitch
        for i in range(len_pitch):
            if pitch_values[i] !=0 and i > i_fin:
                # L'indice de début peut directement être calculé selon
                # le nombre de valeurs en sortie.
                i_output_debut = int(k*i)
                # L'indice de fin doit être calculé selon le nombre de valeurs
                # en entrée.
                i_fin = i
                while pitch_values[i_fin] != 0:
                    i_fin += 1
                # On revient en arrière d'un incrément car on veut que
                # le moment correspondant à i_fin soit parlé.
                i_fin -= 1
                # L'indice de fin est finalement converti selon le nombre de
                # valeurs en sortie.
                i_output_fin = int(k*i_fin)
                if i_output_fin > i_output_debut:
                    moments_parles.append([i_output_debut, i_output_fin])
    
    else :
        k = len_pitch/len_output # Rapport > 1
        
        # On s'assure que la boucle while termine.
        pitch_values[int(-k)] = 0
        
        for i in range(len_output):
            
            pitch_extract = pitch_values[int(k*i):int(k*(i+1))]
            # On regarde sur des périodes de k frames si quelqu'un parle ou non
            if isSpeakingInThisExtract(pitch_extract) and i > i_fin:
                i_debut = i
                i_fin = i_debut
                # Si quelqu'un parle, on regarde sur les périodes suivantes s'il parle toujours
                while isSpeakingInThisExtract(pitch_extract):
                    i_fin += 1
                    pitch_extract = pitch_values[int(k*i_fin):int(k*(i_fin+1))]
                # On revient en arrière d'un incrément car on veut que
                # le moment correspondant à i_fin soit parlé.
                i_fin -= 1
                if i_fin > i_debut:
                    moments_parles.append([i_debut, i_fin])
        
    return moments_parles

## Fonction pour l'affichage du pitch et de l'intensité pour la soutenance finale

def draw_pitch_and_intensisty(filename, title):
    filepath = os.path.join(data_dir, filename + '.wav')
    snd = parselmouth.Sound(filepath)
    
    plt.figure()
    
    # Plot the pitch contour
    plt.subplot(2, 1, 1)
    plt.title(title)
    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    # plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])
    # plt.show() # or plt.savefig("spectrogram_0.03.pdf")
    
    # Plot the intensity
    plt.subplot(2, 1, 2)
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    # plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show() # or plt.savefig("spectrogram.pdf")


# %% Main

sns.set() # Use seaborn's default style to make attractive graphs

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filenames_MJPM = ['MJPM-1','MJPM-2','MJPM-3']
filenames_yuko = ['yuko1','yuko2','yuko3']
names_list = np.array([filenames_MJPM, filenames_yuko])
filepaths = []

for filenames in names_list:
    filepaths.append([os.path.join(data_dir, filename + '.wav') for filename in filenames])

filepath = filepaths[0][0]
snd = parselmouth.Sound(filepath)

# %% Analyse pour l'API avec les fichiers de 1h
data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filename = 'Skilder_190724_TV_MJPM'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)

# %% Plot the sound wave of the audio file.

# Plot nice figures using Python's "standard" matplotlib library
plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show() # or plt.savefig("sound.png")

# %% Plot the spectrogram with the intensity of the sound

intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_intensity(intensity)
plt.xlim([snd.xmin, snd.xmax])
plt.show() # or plt.savefig("spectrogram.pdf")

# %% Plot the pitch contour

pitch = snd.to_pitch()
# If desired, pre-emphasize the sound fragment before calculating the spectrogram
pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show() # or plt.savefig("spectrogram_0.03.pdf")

# %% Plot intensity for each audio file.

plt.figure()

for i in range(6):
    nb_col = len(filepaths[0])                  #3
    filepath = filepaths[i//nb_col][i%nb_col]   #[0][0], [0][1], [0][2]... [1][2]
    
    snd = parselmouth.Sound(filepath)
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    
    plt.subplot(2, 3, i+1)
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])

plt.show()

# %% Plot pitch for each audio file.

plt.figure()

for i in range(6):
    nb_col = len(filepaths[0])                  #3
    filepath = filepaths[i//nb_col][i%nb_col]   #[0][0], [0][1], [0][2]... [1][2]
    
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch()
    # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    
    plt.subplot(2, 3, i+1)
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])

plt.show()

# %% Plot the pitch with an histogram on the right
for i in range(6):
    nb_col = len(filepaths[0])                  #3
    filepath = filepaths[i//nb_col][i%nb_col]   #[0][0], [0][1], [0][2]... [1][2]
    
    snd = parselmouth.Sound(filepath)
    filename = path_show_ext(filepath)[1]
    title = "Pitch et histogramme de l'audio " + filename
    draw_pitch_histogram(snd, title) # C'est ici que ça change

# %% Analyse multilocuteurs

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filenames_MJPM = ['MJPM-1','MJPM-2','MJPM-3']
filenames_MJFG = ['MJFG-1','MJFG-2','MJFG-3']
filenames_MAB = ['MAB-1','MAB-2','MAB-3','MAB-4']
filenames_yuko = ['yuko1','yuko2','yuko3']
speaker_labels = ['MJPM', 'MJFG', 'MAB', 'Yuko']

# Entrainement par le calcul des caractéristiques fréquentielles des locuteurs
f_caracs_yuko = entrainement_reco_pitch(filenames_yuko)
f_caracs_MJPM = entrainement_reco_pitch(filenames_MJPM)
f_caracs_MJFG = entrainement_reco_pitch(filenames_MJFG)
f_caracs_MAB = entrainement_reco_pitch(filenames_MAB)

filename_test = 'Skilder_190724_TV_12-14min'

speaker_ids, scores = diarization(filename_test, f_caracs_yuko, f_caracs_MJPM, f_caracs_MJFG, f_caracs_MAB)

# %% Visualisation des caracs de chaque locuteur pour la démo

fig, ax = plt.subplots()
ax.broken_barh([(5, 10)], (f_caracs_yuko[0], f_caracs_yuko[1]-f_caracs_yuko[0]), facecolors='tab:blue')
ax.broken_barh([(5, 10)], (f_caracs_yuko[2], 5), facecolors='tab:red')
ax.broken_barh([(20, 10)], (f_caracs_MJPM[0], f_caracs_MJPM[1]-f_caracs_MJPM[0]), facecolors='tab:blue')
ax.broken_barh([(20, 10)], (f_caracs_MJPM[2], 5), facecolors='tab:red')
ax.broken_barh([(35, 10)], (f_caracs_MJFG[0], f_caracs_MJFG[1]-f_caracs_MJFG[0]), facecolors='tab:blue')
ax.broken_barh([(35, 10)], (f_caracs_MJFG[2], 5), facecolors='tab:red')
ax.broken_barh([(50, 10)], (f_caracs_MAB[0], f_caracs_MAB[1]-f_caracs_MAB[0]), facecolors='tab:blue')
ax.broken_barh([(50, 10)], (f_caracs_MAB[2], 5), facecolors='tab:red')

plt.title("Caractéristiques fréquentielles des locuteurs")
ax.set_xlim(0, 65)
ax.set_ylim(0, 350)
ax.set_xticks([10, 25, 40, 55])
ax.set_xticklabels(['Yuko', 'MJPM', 'MJFG', 'MAB'])
ax.set_ylabel('Fréquences [Hz]')
plt.show()

# %% Utilisation de la LPC pour déterminer le locuteur

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filename = 'MJPM-1'
filepath = os.path.join(data_dir, filename + '.wav')

snd = parselmouth.Sound(filepath)
spectrum = snd.to_spectrum()
lpc_spectrum = spectrum.lpc_smoothing()
lpc_spectrogram = lpc_spectrum.to_spectrogram()

# %% ACP des pitchs moyens des locuteurs

tableau_donnees, tableaux_locuteurs = pitch_moyen_extraits_parles(filenames_MJPM, filenames_MJFG, filenames_MAB,filenames_yuko)
eigenvalues, eigenvectors = ACP(tableau_donnees)
plot_nuage_individus(eigenvectors, tableaux_locuteurs, speaker_labels)

# %% ACP sur les fréquences de pitch séparées de 5 Hz

pitch_values_MJPM = get_pitch_from_speaker(filenames_MJPM)
title  = "Histogramme des valeurs de pitchs non nulles de MJPM"
hist_MJPM, bin_edges = pitch_values_to_bins(pitch_values_MJPM, plot=True, title=title)

tableau_pitchs_speakers, bins = filenames_to_pitch_hist(filenames_MJPM, filenames_MJFG, filenames_MAB, filenames_yuko, plot=True)
eigenvalues, eigenvectors = ACP(tableau_pitchs_speakers)
plot_nuage_individus(eigenvectors, tableau_pitchs_speakers, speaker_labels)

# %% Plots pour la démo
# Plots de pitch et d'intensité
data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filename = 'yuko1'
title = "Enregistrement de femme"

draw_pitch_and_intensisty(filename, title)

filename = 'MJFG-3'
title = "Enregistrement d'homme"

draw_pitch_and_intensisty(filename, title)

# %% Plots pour la démo
# Plots de pitch avec histogramme sur le côté
filename = 'yuko1'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = "Enregistrement de femme"
draw_pitch_histogram(snd, title)

filename = 'MJFG-3'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = "Enregistrement d'homme"
draw_pitch_histogram(snd, title)

# %% Plots pour la démo
# Plot simple du pitch
filename = 'yuko3'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
pitch = snd.to_pitch()
pitch_values = pitch.selected_array['frequency']
title = "Séparation des pitchs"

plt.figure()
x = np.linspace(0, len(pitch_values)/100, num=len(pitch_values))
plt.plot(x, pitch_values)
plt.xlabel("time [s]")
plt.ylabel("frequency [Hz]")
plt.title(title)

# %% Plots pour la démo
# Plots des histogrammes des extraits de pitch

filename = 'yuko3-1'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = 'Extrait de pitch 1'
draw_pitch_histogram(snd, title)

filename = 'yuko3-2'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = 'Extrait de pitch 2'
draw_pitch_histogram(snd, title)

filename = 'yuko3-3'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = 'Extrait de pitch 3'
draw_pitch_histogram(snd, title)

filename = 'yuko3-3pitchs'
filepath = os.path.join(data_dir, filename + '.wav')
snd = parselmouth.Sound(filepath)
title = 'Les trois extraits de pitchs'
draw_pitch_histogram(snd, title)