"""!  
@brief Example 02 
@details Example of spectrogram computation for a wav file, using only scipy 
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
@adapted by Raphaël Chalicarne {raphael.chalicarne@ecl17.ec-lyon.fr}
""" 

# %% Imports

import scipy.fftpack as scp 
import numpy as np 
import scipy.io.wavfile as wavfile 
import plotly 
import plotly.graph_objs as go 
#import aubio
import os

# %% Set the input audio file.

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filename = 'Skilder_190724_TV_MJPM_1_min'
input_show = os.path.join(data_dir, filename + '.wav')

# %% Parameters

win = 0.02

# %% Functions

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

def get_fft_spec(signal, fs, win): 
    frame_size, signal_len, spec, times = int(win * fs), len(signal), [], [] 
    # break signal into non-overlapping short-term windows (frames) 
    frames = np.array([signal[x:x + frame_size] for x in 
                       np.arange(0, signal_len - frame_size, frame_size)]) 
    for i_f, f in enumerate(frames): # for each frame 
        times.append(i_f * win) 
        # append mag of fft 
        X = np.abs(scp.fft(f)) ** 2 
        freqs = np.arange(0, 1, 1.0/len(X)) * (fs/2) 
        spec.append(X[0:int(len(X)/2)] / X.max())
    
    S = np.array(spec).T
    time_len = int(signal_len/frame_size) -1
    freq_incr = 1.0/len(X) * (fs/2)
    # signal_max gives an array of the frequency with the maximum amplitude at 
    # each time increment.
    signal_max = np.array([S[:,x].argmax()*freq_incr for x in range(time_len)])
    
    return np.array(spec).T, freqs, times, signal_max

def max_signal(signal, fs, win):
    S, f, t = get_fft_spec(signal, fs, win)
    frame_size, signal_len = int(win * fs), len(signal)
    time_len = int(signal_len/frame_size) -1
    signal_max = np.array([S[:,x].argmax() for x in range(time_len)])
    return signal_max

def classify_voices(signal_max):
    man_voices = [freq if freq < 200 else 0 for freq in signal_max]
    woman_voices = [freq if 200 <= freq < 400 else 0 for freq in signal_max]
    else_voices = [freq if freq >= 400 else 0 for freq in signal_max]
    return man_voices, woman_voices, else_voices

def barh_voices(voice_array, win):
    #voice_array = np.array(voice_array)
    non_zeros_indices = np.nonzero(voice_array)[0]
    x_list = []
    for i in non_zeros_indices:
        x_list.append([i, i, i+1, i+1])
    x_array = win*np.array(x_list)
    return x_array

# %% Main
    # Fs : int, Sample rate of wav file
    # s : numpy array, Data read from wav file
    
# %% Spectrogramme de l'extrait audio
if __name__ == '__main__':
    [Fs, s] = wavfile.read(input_show)
    S, f, t, s_m = get_fft_spec(s, Fs, win) 
    heatmap = go.Heatmap(z=S, y=f, x=t)
    layout_spectrogram = go.Layout(title='Spectrogram Calculation Example', 
                                  xaxis=dict(title='time (sec)',), 
                                  yaxis=dict(title='Freqs (Hz)',)) 
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout_spectrogram), 
                      filename="temp.html", auto_open=True)
    
# %% Spectre de l'incrément à t = 0,2 s
if __name__ == '__main__':
    time = 0.2
    i_t = int(time/win)
#    scatter = go.Scatter(x=f, y=S[:,i_t], name="linear", line_shape='linear')
#    layout_frequency = go.Layout(title='Spectre en fréquence à t = ' + str(time), 
#                                  xaxis=dict(title='Freqs (Hz)',), 
#                                  yaxis=dict(title='Amplitude',)) 
    
#    plotly.offline.plot(go.Figure(data=[scatter],
#                                    layout=layout_frequency),
#                                    filename="temp2.html",
#                                    auto_open=True)

# %% Affichage des harmoniques à chaque incrément (scatter plot)
if __name__ == '__main__':
    scatter_harmonique = go.Scatter(x=t, y=s_m, name="linear", line_shape='linear')
    layout_filtered_frequency = go.Layout(title='Harmoniques', 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Freqs (Hz)',)) 
#    plotly.offline.plot(go.Figure(data=[scatter_harmonique],
#                                    layout=layout_filtered_frequency),
#                                    filename="temp3.html",
#                                    auto_open=True)
    
    ## Classification des voix selon les fréquences des harmoniques de chaque
    ## incrément.
    man_voices, woman_voices, else_voices = classify_voices(s_m)
    scatter_hommes = go.Scatter(x=t, y=man_voices, name="Man voices", mode="markers")
    scatter_femmes = go.Scatter(x=t, y=woman_voices, name="Woman voices", mode="markers")
    scatter_autres = go.Scatter(x=t, y=else_voices, name="Other sounds", mode="markers")
    layout_classified_voices = go.Layout(title='Voix classifiées selon la bande de fréquence', 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Freqs (Hz)',)) 
#    plotly.offline.plot(go.Figure(data=[scatter_hommes, scatter_femmes, scatter_autres],
#                        layout=layout_classified_voices),
#                        filename="temp4.html",
#                        auto_open=True)
    
# %% Affichage des incréments classés sous forme de barres verticales.

if __name__ == '__main__':
    # Parameters
    y_wide = 0.6
    colors = ['blue','red','green']
    
    [x_men, x_women, x_else] = [barh_voices(voices, win) for voices in [man_voices, woman_voices, else_voices]]
    x_arrays = [x_men, x_women, x_else] 
    
    data_x = []
    for i, x_array in enumerate(x_arrays):
        
        data_x+=[go.Scatter(x = frame,
                       y = [i-y_wide/2, i+y_wide/2, i+y_wide/2, i-y_wide/2],
                       fill='toself',
                       fillcolor = colors[i],
                       mode='none')
                       for frame in x_array]
    layout_speakers = go.Layout(title='Séparation des locuteurs basé sur les bandes de fréquence des harmoniques', 
                                          xaxis=dict(title='time (sec)',), 
                                          yaxis=dict(title='Freqs (Hz)',)) 
    plotly.offline.plot(go.Figure(data = data_x,
                        layout=layout_speakers),
                        filename="temp5.html",
                        auto_open=True)
    
# %% Spectre de l'extrait audio

data_dir = os.path.join('Support_CentraleDigitale_Lab_201920', 'Data_Submarin', 'Dataset_J1')
filenames_homme = ['homme1','homme2','homme3']
filenames_yuko = ['yuko1','yuko2','yuko3']
filenames = np.array([filenames_homme, filenames_yuko])

# %% Spectre hommes + Yuko
signals = []
Y_inputs = []
frequences = []
scatter_spectres = []
colors = ['blue','red']
i_audio = 0

if __name__ == '__main__':
    for i, names_list in enumerate(filenames):
        audio_inputs = [os.path.join(data_dir, filename + '.wav') for filename in names_list]
        for audio_input in audio_inputs:
            [Fs, s] = wavfile.read(audio_input)
            signals.append([Fs, s])
            Y_inputs.append(np.abs(scp.rfft(s)) ** 2)
            frequences.append(np.arange(0, 1, 1.0/len(Y_inputs[i_audio])) * (Fs/2)/2)
            basename = path_show_ext(audio_input)[1]
            scatter_spectres.append(go.Scatter(x=frequences[i_audio],
                                               y=Y_inputs[i_audio]/Y_inputs[i_audio].max(),
                                               name=basename,
                                               line_shape='linear',
                                               line_color = colors[i],
                                               opacity=0.5
                                               )
                                    )
            i_audio+=1
        
    layout_spectre_audio_input = go.Layout(title="Spectre des extraits du même homme et de Yuko", 
                                          xaxis=dict(title='Freqs (Hz)',), 
                                          yaxis=dict(title='Intensité normalisée',))
    
#    plotly.offline.plot(go.Figure(data = scatter_spectres,
#                        layout=layout_spectre_audio_input),
#                        filename="temp_scatter_hommes_yuko.html",
#                        auto_open=True)
    
# %% Analyse des pitchs avec Aubio
# Documentation : https://aubio.org/manual/latest/py_io.html
    
#if __name__ == '__main__':
#    src = aubio.source(input_show)
    

