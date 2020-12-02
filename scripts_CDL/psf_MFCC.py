# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:38:10 2019
@author: Raphaël Chalicarne
"""
# %% Importations

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import parselmouth

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly 
import plotly.graph_objs as go

# %% Fonctions

def get_mfcc_means(filename, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    filepath = os.path.join(data_dir, filename + '.wav')
    (rate,sig) = wav.read(filepath)
    # exclude 0th order coefficient (much larger than others)
    mfcc_feat = mfcc(sig,rate,nfft = 2048, winfunc=np.hamming)[:,1:]
    mfcc_means = np.mean(mfcc_feat, axis=0)
    
    return mfcc_means

def plot_mfcc_means(locuteurs_mfcc_means):
    plt.figure()
    for mfcc_means in locuteurs_mfcc_means:
        plt.plot(mfcc_means)

def get_mfcc_from_files(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    all_mfcc_feat = []
    for filename in filenames:
        filepath = os.path.join(data_dir, filename + '.wav')
        (rate,sig) = wav.read(filepath)
        # exclude 0th order coefficient (much larger than others)
        mfcc_feat = mfcc(sig,rate,nfft = 2048, winfunc=np.hamming)[:,1:]
        all_mfcc_feat.append(mfcc_feat)
    
    mfcc_concatenated = np.concatenate(tuple(all_mfcc_feat), axis = 0)
    return mfcc_concatenated

def get_mfcc_means_from_files(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    mfcc_concatenated = get_mfcc_from_files(filenames, data_dir)
    mfcc_means = np.mean(mfcc_concatenated, axis  = 0)
    return mfcc_means

def mfcc_for_plot(filenames,
                  data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1'),
                  color='r',
                  mfcc_indices = np.linspace(2,13,12),
                  plot=True):
    """Input :
        Liste des noms de fichiers audio d'un même locuteur
    Output:
        mfcc_means : Liste des coefs MFCC de chaque audio
        mfcc_max : Liste des coefs max des listes de mfcc_means
        mfcc_min : Liste des coefs min des listes de mfcc_means
        mfcc_total_mean : Moyenne des coefs MFCC pour tous les enregistrements combinés""" 
    mfcc_means = np.array([get_mfcc_means(file_locuteur) for file_locuteur in filenames])
    mfcc_max = np.max(mfcc_means, axis = 0)
    mfcc_min = np.min(mfcc_means, axis = 0)
    mfcc_total_mean = get_mfcc_means_from_files(filenames)
    
    # Plot
    if plot:
        plt.figure()
        plt.fill_between(mfcc_indices, mfcc_min, mfcc_max, color=color, alpha=0.5)
        plt.scatter(mfcc_indices, mfcc_total_mean, color = color)
        plt.title("Coeffcients MFCC de " + filenames[0])
    
    return mfcc_means, mfcc_max, mfcc_min, mfcc_total_mean    

def pitch_clusterisation(filename,
                         data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
                         ):
    """Input : a parselmouth sound
    Output : Liste des indices de début de pitch
    Extraits de pitch"""
    filepath = os.path.join(data_dir, filename + '.wav')
    snd = parselmouth.Sound(filepath)
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

# Fonctions reconnaissance du locuteur
def distance_mfcc(mfcc_speaker_means, mfcc_test_means):
    """Input : Two arrays of 13 elements
    Output : The sum of the difference bewteen each coefficient"""
    return np.sum(np.abs(mfcc_speaker_means - mfcc_test_means))

def mfcc_speaker_recognition(filename, *filenames_speakers, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    indices_debut, p = pitch_clusterisation(filename, data_dir)
    
    filepath = os.path.join(data_dir, filename + '.wav')
    (rate,sig) = wav.read(filepath)
    
    mfcc_speakers_carac = []
    data_speaker = []
    # f_speakers = ['MJPM-1','MJPM-2','MJPM-3'] par exemple
    for f_speakers in filenames_speakers:
        mfcc_means, mfcc_max, mfcc_min, mfcc_total_mean = mfcc_for_plot(f_speakers, data_dir, plot=False)
        label_speaker = f_speakers[0]
        mfcc_speakers_carac.append([label_speaker, mfcc_means, mfcc_max, mfcc_min, mfcc_total_mean])
    
    nb_speakers = len(mfcc_speakers_carac)
    nb_extracts = len(p)
    
    scores_diff = np.zeros((nb_extracts, nb_speakers), float)
    
    for i, p_extract in enumerate(p):
        indice_debut_extract = indices_debut[i]
        len_extract = len(p_extract)
        # Extrait du signal dont on va récupérer les MFCC
        signal_extract = sig[indice_debut_extract:indice_debut_extract+len_extract]
        # exclude 0th order coefficient (much larger than others)
        mfcc_feat_extract = mfcc(signal_extract,rate,nfft = 2048, winfunc=np.hamming)[:,1:]
        mfcc_extract_means = np.mean(mfcc_feat_extract, axis=0)
        
        # mfcc_speaker_carac[id_speaker] = [label_speaker, mfcc_means, mfcc_max, mfcc_min, mfcc_total_mean]
        for id_speaker in range(nb_speakers):
            label_speaker = mfcc_speakers_carac[id_speaker][0]
            mfcc_total_mean_speaker = mfcc_speakers_carac[id_speaker][4]
            diff_speaker_extract = distance_mfcc(mfcc_total_mean_speaker, mfcc_extract_means)
            scores_diff[i, id_speaker] = diff_speaker_extract
    
    # Plot the result
        points_x = np.array([indices_debut[i],
                    indices_debut[i],
                    indices_debut[i] + len(p_extract),
                    indices_debut[i] + len(p_extract)])/100
    
        id_speaker_extract = np.argmin(scores_diff[i])
        
        points_y = [id_speaker_extract - 1/2, id_speaker_extract + 1/2, id_speaker_extract + 1/2, id_speaker_extract - 1/2]
    
        data_speaker+=[go.Scatter(x = points_x,
                              y = points_y,
                              fill='toself',
                              fillcolor = 'blue',
                              mode = 'none')]
    layout_speakers = go.Layout(title='Séparation des locuteurs basé sur les MFCC, extrait ' + filename, 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Locuteurs : MJPM - MJFG - MAB - Yuko',))
    plotly.offline.plot(go.Figure(data = data_speaker,
                                layout=layout_speakers),
                                filename= "mfcc_separation_" + filename+".html",
                                auto_open=True)
    return scores_diff

# MFCC basés sur les extraits parlés
def mfcc_locuteur_speaking(filenames, data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    # filenames = ['MJPM-1','MJPM-2','MJPM-3'] par exemple
    # f_speaker = 'MJPM-1' par exemple
    all_mfcc_speaker = []
    
    for f_speaker in filenames:
        # On récupère les indices de chaque extrait parlé
        # Ainsi que la longueur de chaque extrait
        indices_debut, p = pitch_clusterisation(f_speaker, data_dir)
        
        filepath = os.path.join(data_dir, f_speaker + '.wav')
        (rate,sig) = wav.read(filepath)
        
        # On parcourt les extraits parlés de chaque fichier f_speaker
        for i, p_extract in enumerate(p):
            indice_debut_extract = indices_debut[i]
            len_extract = len(p_extract)
            sig_extract = sig[indice_debut_extract:indice_debut_extract+len_extract]
            # exclude 0th order coefficient (much larger than others)
            mfcc_feat = mfcc(sig_extract,rate,nfft = 2048, winfunc=np.hamming)[:,1:]
            all_mfcc_speaker.append(mfcc_feat)
    
    mfcc_concatenated = np.concatenate(tuple(all_mfcc_speaker), axis = 0)
    return mfcc_concatenated

def plot_mfcc_locuteur_speaking(speaker_labels, *mfcc_locuteurs_speaking, mfcc_indices = np.linspace(2,13,12), title=None):
    """Input : Arrays given by the function mfcc_locuteur_speaking
    Output : Plot of the min, max, and mean value of each mfcc coefficient of each speaker"""
    colors = ['black', 'b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    for i, mfcc_locuteur_speaking in enumerate(mfcc_locuteurs_speaking):
        mfcc_loc_means = np.mean(mfcc_locuteur_speaking, axis =0)
        mfcc_loc_max = np.max(mfcc_locuteur_speaking, axis = 0)
        mfcc_loc_min = np.min(mfcc_locuteur_speaking, axis = 0)
        
        plt.fill_between(mfcc_indices, mfcc_loc_min, mfcc_loc_max, color=colors[i], alpha=0.5)
        plt.scatter(mfcc_indices, mfcc_loc_means, color = colors[i], label=speaker_labels[i])
    plt.legend()
    plt.title(title)
    
def plot_mfcc_locuteur_speaking_scatter(speaker_labels, *mfcc_locuteurs_speaking, mfcc_indices = np.linspace(2,13,12), title=None):
    """Input : Arrays given by the function mfcc_locuteur_speaking
    Output : Plot of the min, max, and mean value of each mfcc coefficient of each speaker"""
    colors = ['black', 'b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    for i, mfcc_locuteur_speaking in enumerate(mfcc_locuteurs_speaking):
        for mfcc_extract in mfcc_locuteur_speaking:
            plt.scatter(mfcc_indices, mfcc_extract, color = colors[i], alpha=0.5, label=speaker_labels[i])
    plt.legend()
    plt.title(title)

def get_24_coefficients(filenames, data_dir=os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')):
    """Input : audio recordings filenames list of a same speaker
    For instance, filenames_MJPM = ['MJPM-1','MJPM-2','MJPM-3']
    Output : Array with 24 columns corresponding to the MFC Coefficients 2-13, 
    and the delta MFC Coefficients of these MFC Coefficients.
    It has I rows corresponding to speaking frames of the recordings."""
    # The 12 MFC Coefficients 2-13
    mfcc_speaker_speaking = mfcc_locuteur_speaking(filenames, data_dir)
    # The 12 delta MFC Coefficients corresponding to the previous coefficients
    d_mfcc_speaker_speaking = delta(mfcc_speaker_speaking, 2)
    mfcc_24_coeffs = np.hstack((mfcc_speaker_speaking, d_mfcc_speaker_speaking))
    return mfcc_24_coeffs
    
        
# Fonctions ACP
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

def coords_nuage_variables(eigenvectors, rayon):
    """Input : Vecteurs propres issus de l'ACP
    Output : Coordonnées des variables pour qu'elles puissent être affichées sur le nuage des variables"""
    module = np.sqrt(eigenvectors[0]**2 + eigenvectors[1]**2)
    coords = rayon*np.divide(eigenvectors[:2], module, out=np.zeros_like(eigenvectors[:2]), where=module!=0)
    return coords

def plot_nuage_individus(eigenvectors, speaker_labels, *all_mfcc_locuteurs, plot_variables=False, title=None):
    colors = ['b', 'r', 'g', 'black', 'c', 'm', 'y', 'k', 'w']
    
    if plot_variables:
        nb_variables = len(eigenvectors)
        max_F1, min_F1, max_F2, min_F2 = 0, 0, 0, 0
    
    plt.figure()
    for i, all_mfcc_locuteur in enumerate(all_mfcc_locuteurs):
        coor_F1_locuteur = all_mfcc_locuteur.dot(eigenvectors[0])
        coor_F2_locuteur = all_mfcc_locuteur.dot(eigenvectors[1])
        plt.scatter(coor_F1_locuteur, coor_F2_locuteur, color = colors[i], alpha=0.5, label=speaker_labels[i])
    
        if plot_variables:
            # Calcul des limites du cercle des variables
            min_F1 = np.min((min_F1,np.min(coor_F1_locuteur)))
            max_F1 = np.max((max_F1,np.max(coor_F1_locuteur)))
            min_F2 = np.min((min_F2,np.min(coor_F2_locuteur)))
            max_F2 = np.max((max_F2,np.max(coor_F2_locuteur)))
    
    if plot_variables:
        # Tracer le cercle des variables
        rayon = 0.75*np.abs(np.min([min_F1,max_F1,min_F2,max_F2]))
        coords_variables = coords_nuage_variables(eigenvectors, rayon)
        an = np.linspace(0, 2*np.pi, 100)
        plt.plot(rayon * np.cos(an), rayon * np.sin(an), color='black')
        for i in range(nb_variables):
            plt.text(1.1 * coords_variables[0,i], 1.1 * coords_variables[1,i], str(i+1))
        plt.scatter(coords_variables[0], coords_variables[1], color='black', s=100)
    
    str_title = "Projection du nuage des individus sur F1 et F2" + "\n" + title
    plt.title(str_title)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()

# %% Importation du fichier

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filenames_MJPM = ['MJPM-1','MJPM-2','MJPM-3']
filenames_MJFG = ['MJFG-1','MJFG-2','MJFG-3']
filenames_MAB = ['MAB-1','MAB-2','MAB-3','MAB-4']
filenames_yuko = ['yuko1','yuko2','yuko3']
speaker_labels = ['MJPM', 'MJFG', 'MAB', 'Yuko']

filenames_speakers = [filenames_MJPM, filenames_MJFG, filenames_MAB, filenames_yuko]

filename = 'MAB-1'
filepath = os.path.join(data_dir, filename + '.wav')

# %% Main

(rate,sig) = wav.read(filepath)
# exclude 0th order coefficient (much larger than others)
mfcc_feat = mfcc(sig,rate,nfft = 2048, winfunc=np.hamming)[:,1:]

snd = parselmouth.Sound(filepath)

# Histogram
# now determine nice limits by hand:
binwidth = 1
lim_max = np.ceil(mfcc_feat.max() / binwidth) * binwidth
lim_min = np.ceil(mfcc_feat.min() / binwidth) * binwidth

bins = np.arange(lim_min, lim_max + binwidth, binwidth)

mfcc_medians = np.median(mfcc_feat, axis=0)
mfcc_means = np.mean(mfcc_feat, axis=0)

# %% Pour un locuteur

# Abcisses
mfcc_indices = np.linspace(2,13,12)

# Moyenne des coefs MFCC pour chaque enregistrement
MJPM_mfcc_means = np.array([get_mfcc_means(file_locuteur) for file_locuteur in filenames_MJPM])
MJFG_mfcc_means = np.array([get_mfcc_means(file_locuteur) for file_locuteur in filenames_MJFG])
MAB_mfcc_means = np.array([get_mfcc_means(file_locuteur) for file_locuteur in filenames_MJPM])
yuko_mfcc_means = np.array([get_mfcc_means(file_locuteur) for file_locuteur in filenames_yuko])

MJPM_max = np.max(MJPM_mfcc_means, axis = 0)
MJPM_min = np.min(MJPM_mfcc_means, axis = 0)

# Moyenne des coefs MFCC pour tous les enregistrements combinés
MJPM_mfcc_total_mean = get_mfcc_means_from_files(filenames_MJPM)

#fig, (ax1, ax2, ax3) = plt.plot()
#plt.figure()
#fig, ax1 = plt.subplots(1, 1)
#
#ax1.fill_between(mfcc_indices, MJPM_min, MJPM_max, color='r', alpha=0.5)
#ax1.scatter(mfcc_indices,MJPM_mfcc_total_mean, color = 'r')

MJPM_mfcc_means, MJPM_mfcc_max, MJPM_mfcc_min, MJPM_mfcc_total_mean = mfcc_for_plot(filenames_MJPM, color='b')
MJFG_mfcc_means, MJFG_mfcc_max, MJFG_mfcc_min, MJFG_mfcc_total_mean = mfcc_for_plot(filenames_MJFG, color='r')
MAB_mfcc_means, MAB_mfcc_max, MAB_mfcc_min, MAB_mfcc_total_mean = mfcc_for_plot(filenames_MAB, color='g')
yuko_mfcc_means, yuko_mfcc_max, yuko_mfcc_min, yuko_mfcc_total_mean = mfcc_for_plot(filenames_yuko, color='black')

# %% Plot sur une même figure
plt.figure()

plt.fill_between(mfcc_indices, MJPM_mfcc_min, MJPM_mfcc_max, color='b', alpha=0.5)
plt.fill_between(mfcc_indices, MJFG_mfcc_min, MJFG_mfcc_max, color='r', alpha=0.5)
plt.fill_between(mfcc_indices, MAB_mfcc_min, MAB_mfcc_max, color='g', alpha=0.5)
plt.fill_between(mfcc_indices, yuko_mfcc_min, yuko_mfcc_max, color='black', alpha=0.5)

plt.scatter(mfcc_indices, MJPM_mfcc_total_mean, color = 'b', label='MJPM')
plt.scatter(mfcc_indices, MJFG_mfcc_total_mean, color = 'r', label='MJFG')
plt.scatter(mfcc_indices, MAB_mfcc_total_mean, color = 'g', label='MAB')
plt.scatter(mfcc_indices, yuko_mfcc_total_mean, color = 'black', label='yuko')

plt.title("Coefficients MFCC de différents locuteurs")
plt.legend()

# %% Reconnaissance du locuteur

scores_diff = mfcc_speaker_recognition('Skilder_190724_TV_12-14min', filenames_MJPM, filenames_MJFG, filenames_MAB, filenames_yuko)

# %% Aanlyse en composantes principales (ACP) basée sur tous les extraits MFCC
# de chaque enregistrement (inclus les silences)

all_mfcc_MJPM = get_mfcc_from_files(filenames_MJPM)
all_mfcc_MJFG = get_mfcc_from_files(filenames_MJFG)
all_mfcc_MAB = get_mfcc_from_files(filenames_MAB)
all_mfcc_yuko = get_mfcc_from_files(filenames_yuko)

all_mfcc = np.concatenate((all_mfcc_MJPM, all_mfcc_MJFG, all_mfcc_MAB, all_mfcc_yuko), axis = 0)
eigenvalues, eigenvectors = ACP(all_mfcc)

# Projection du nuage des individus
plot_nuage_individus(eigenvectors, speaker_labels, all_mfcc_MJPM, all_mfcc_MJFG, all_mfcc_MAB, all_mfcc_yuko)

# %% Projection du nuage des variables
plt.figure()
#plt.scatter(eigenvectors[0],eigenvectors[1])

for i in range(2,13):
    plt.arrow(0, 0, eigenvectors[0,i], eigenvectors[1,i], head_width=0.05, head_length=0.1, fc='k', ec='k')

# %% ACP sur les moyennes des coefficients MFCC de chaque locuteur
    
MJPM_mfcc_means, MJPM_mfcc_max, MJPM_mfcc_min, MJPM_mfcc_total_mean = mfcc_for_plot(filenames_MJPM, color='b', plot=False)
MJFG_mfcc_means, MJFG_mfcc_max, MJFG_mfcc_min, MJFG_mfcc_total_mean = mfcc_for_plot(filenames_MJFG, color='r', plot=False)
MAB_mfcc_means, MAB_mfcc_max, MAB_mfcc_min, MAB_mfcc_total_mean = mfcc_for_plot(filenames_MAB, color='g', plot=False)
yuko_mfcc_means, yuko_mfcc_max, yuko_mfcc_min, yuko_mfcc_total_mean = mfcc_for_plot(filenames_yuko, color='black', plot=False)

# Tableau sur lequel on va appliquer l'ACP
mfcc_total_means_concatenated = np.vstack((MJPM_mfcc_total_mean,
                                                MJFG_mfcc_total_mean,
                                                MAB_mfcc_total_mean,
                                                yuko_mfcc_total_mean))

eigenvalues, eigenvectors = ACP(mfcc_total_means_concatenated)

# Projection du nuage des individus
plot_nuage_individus(eigenvectors, speaker_labels, all_mfcc_MJPM, all_mfcc_MJFG, all_mfcc_MAB, all_mfcc_yuko, speaker_labels)

# %% ACP sur les coefficients MFCC des extraits parlés de chaque locuteur

# Abcisses
mfcc_indices = np.linspace(2,13,12)

# Coefficients de chaque locuteur au moment où ils parlent
mfcc_MJPM_speaking = mfcc_locuteur_speaking(filenames_MJPM)
mfcc_MJFG_speaking = mfcc_locuteur_speaking(filenames_MJFG)
mfcc_MAB_speaking = mfcc_locuteur_speaking(filenames_MAB)
mfcc_yuko_speaking = mfcc_locuteur_speaking(filenames_yuko)

title_MFCC_speaking = "Coeffcients MFCC basés sur les extraits parlés de chaque locuteur"

# Plot des coefficients MFCC de chaque locuteur
plot_mfcc_locuteur_speaking(speaker_labels,
                            mfcc_MJPM_speaking,
                            mfcc_MJFG_speaking,
                            mfcc_MAB_speaking, 
                            mfcc_yuko_speaking,
                            title = title_MFCC_speaking)
plot_mfcc_locuteur_speaking_scatter(speaker_labels, 
                                    mfcc_MJPM_speaking, 
                                    mfcc_MJFG_speaking, 
                                    mfcc_MAB_speaking, 
                                    mfcc_yuko_speaking,
                                    title=title_MFCC_speaking)

# Tableau sur lequel on va appliquer l'ACP
mfcc_locuteur_speaking_concatenated = np.vstack((mfcc_MJPM_speaking,
                                                mfcc_MJFG_speaking,
                                                mfcc_MAB_speaking,
                                                mfcc_yuko_speaking))

eigenvalues, eigenvectors = ACP(mfcc_locuteur_speaking_concatenated)

# Projection du nuage des individus
plot_nuage_individus(eigenvectors,
                     speaker_labels, 
                     mfcc_MJPM_speaking, 
                     mfcc_MJFG_speaking, 
                     mfcc_MAB_speaking, 
                     mfcc_yuko_speaking,
                     title = title_MFCC_speaking)

# Calcul des deltas MFCC de chaque locuteur au moment où ils parlent
d_mfcc_MJPM_speaking = delta(mfcc_MJPM_speaking, 2)
d_mfcc_MJFG_speaking = delta(mfcc_MJFG_speaking, 2)
d_mfcc_MAB_speaking = delta(mfcc_MAB_speaking, 2)
d_mfcc_yuko_speaking = delta(mfcc_yuko_speaking, 2)

title_d_MFCC_speaking = "Coeffcients delta MFCC basés sur les extraits parlés de chaque locuteur"

# Plot des coefficients d_MFCC de chaque locuteur
plot_mfcc_locuteur_speaking(speaker_labels,
                            d_mfcc_MJPM_speaking, 
                            d_mfcc_MJFG_speaking, 
                            d_mfcc_MAB_speaking, 
                            d_mfcc_yuko_speaking,
                            title=title_d_MFCC_speaking)

plot_mfcc_locuteur_speaking_scatter(speaker_labels,
                                    d_mfcc_MJPM_speaking, 
                                    d_mfcc_MJFG_speaking, 
                                    d_mfcc_MAB_speaking, 
                                    d_mfcc_yuko_speaking,
                                    title=title_d_MFCC_speaking)

# Tableau sur lequel on va appliquer l'ACP
d_mfcc_locuteur_speaking_concatenated = np.vstack((d_mfcc_MJPM_speaking,
                                                   d_mfcc_MJFG_speaking,
                                                   d_mfcc_MAB_speaking,
                                                   d_mfcc_yuko_speaking))

d_eigenvalues, d_eigenvectors = ACP(d_mfcc_locuteur_speaking_concatenated)

# Projection du nuage des individus
plot_nuage_individus(d_eigenvectors,
                     speaker_labels, 
                     mfcc_MJPM_speaking, 
                     mfcc_MJFG_speaking, 
                     mfcc_MAB_speaking, 
                     mfcc_yuko_speaking,
                     title = title_d_MFCC_speaking)

# %% Analyse MFCC et d_MFCC combinés

# 24 Coefficients de chaque locuteur au moment où ils parlent
mfcc_24_MJPM_speaking = get_24_coefficients(filenames_MJPM)
mfcc_24_MJFG_speaking = get_24_coefficients(filenames_MJFG)
mfcc_24_MAB_speaking = get_24_coefficients(filenames_MAB)
mfcc_24_yuko_speaking = get_24_coefficients(filenames_yuko)

title_24_coefs = "12 coefficients MFCC + 12 coefficients delta MFCC"

# Tableau sur lequel on va appliquer l'ACP
mfcc_24_all_speaking = np.vstack((mfcc_24_MJPM_speaking,
                                  mfcc_24_MJFG_speaking,
                                  mfcc_24_MAB_speaking,
                                  mfcc_24_yuko_speaking))

eigenvalues_24, eigenvectors_24 = ACP(mfcc_24_all_speaking)

# Projection du nuage des individus
plot_nuage_individus(eigenvectors_24,
                     speaker_labels, 
                     mfcc_24_MJPM_speaking, 
                     mfcc_24_MJFG_speaking, 
                     mfcc_24_MAB_speaking, 
                     mfcc_24_yuko_speaking,
                     plot_variables=True,
                     title = title_24_coefs)
