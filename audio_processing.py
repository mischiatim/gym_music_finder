# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a module with audioprocessing functions for my music project.

Matteo Mischiati
June 2021
"""

import os
import numpy as np
from pydub import AudioSegment
import librosa 

#Helper function that searches for files of a certain format
def filetype_search(filetype,folder='./'):
    """
    Returns list of files with extension 'filetype' within current folder or in optional 'folder'
    """
    file_list = []

    for item in os.listdir(folder):
        item_split = item.split('.')
        if item_split[-1]==filetype:
            file_list.append(item)
    
    return(file_list)


#Helper function to convert all of an audiosegment (acquired with Pydub from an .mp3 or other format file) 
#or a section of it to .wav 
def export_AudioSegment_to_wav(audiosegment,wav_filename,trim_loc='none',desired_s=30,desired_rate=22050,desired_sample_width=2):
    """
    Takes a PyDub AudioSegment and exports it to .wav format (with a desired sample rate and width) after optionally 
    trimming the audio (if trim_loc='start','mid' or 'end', then a window of 'desired_s' seconds (default 30) 
    is trimmed before exporting)
    
    """    
    rate = audiosegment.frame_rate
    sample_width = audiosegment.sample_width
    if trim_loc=='none':
        if (rate == desired_rate) and (sample_width == desired_sample_width):
            audiosegment.export(wav_filename, format="wav")
        elif (rate == desired_rate) and (sample_width != desired_sample_width):
            audiosegment.set_sample_width(desired_sample_width).export(wav_filename, format="wav")
        elif (rate != desired_rate) and (sample_width == desired_sample_width):
            audiosegment.set_frame_rate(desired_rate).export(wav_filename, format="wav")
        else:
            audiosegment.set_frame_rate(desired_rate).set_sample_width(desired_sample_width).export(wav_filename, format="wav")
    else: 
        desired_n = np.round(desired_s*rate)
        if trim_loc=='start':
            start_frame = 0
            end_frame = desired_n
        elif trim_loc=='end':
            end_frame = int(audiosegment.frame_count())
            start_frame = end_frame-desired_n
        elif trim_loc=='mid':
            start_frame= int(audiosegment.frame_count()/2)-int(desired_n/2)
            end_frame = start_frame + desired_n
        audiosegment_trimmed = audiosegment.get_sample_slice(start_frame,end_frame) 
        if (rate == desired_rate) and (sample_width == desired_sample_width):
            audiosegment_trimmed.export(wav_filename, format="wav")
        elif (rate == desired_rate) and (sample_width != desired_sample_width):
            audiosegment_trimmed.set_sample_width(desired_sample_width).export(wav_filename, format="wav")
        elif (rate != desired_rate) and (sample_width == desired_sample_width):
            audiosegment_trimmed.set_frame_rate(desired_rate).export(wav_filename, format="wav")
        else:
            audiosegment_trimmed.set_frame_rate(desired_rate).set_sample_width(desired_sample_width).export(wav_filename, format="wav")
        

#Function that goes through all the audio files in a given folder (.mp3, .aac, .wma or .wav)
#and exports them to .wav format (with a desired sample rate and width) after optionally 
#trimming all audiofiles to a common duration (if trim_loc='start','mid' or 'end', then a window of 'desired_s' seconds (default 30) is trimmed before exporting)

#First I need a helper function to identify .wav files already converted
def is_converted_wav(wav_file,suffix='_conv'):
    """
    Checks if wav_file ends with suffix wav_suffix, in which case it has already been converted
    """  
    n_suffix_chars = len(suffix)
    wav_filename_no_extension = wav_file.split(sep='.wav')[0]
    return (len(wav_filename_no_extension)>n_suffix_chars and wav_filename_no_extension[-n_suffix_chars:] == suffix)

def convert_subfolder_audios_to_wav(folder='./',wav_suffix='_conv',trim_loc='none',desired_s=30,desired_rate=22050,desired_sample_width=2):
    """
    Takes all audio files (.mp3, .aac, .wma, .wav) in folder and exports them to .wav format with a desired sample rate 
    and width) after optionally trimming the audio (if trim_loc='start','mid' or 'end', 
    then a window of 'desired_s' seconds (default 30) is trimmed before exporting)
    
    """   
    
    audiotypes = ['mp3','aac','wma']
    
    wav_list = filetype_search('wav',folder)
    
    for filetype in audiotypes:
        
        extension = '.' + filetype
        
        filelist = filetype_search(filetype,folder)
        
        for file in filelist:
            wav_filename_w_suffix = file.split(sep=extension)[0] + wav_suffix + '.wav'
            
            if wav_filename_w_suffix not in wav_list:
            
                tempaudio = AudioSegment.from_file(folder+file)
            
                export_AudioSegment_to_wav(tempaudio,folder+wav_filename_w_suffix,trim_loc,desired_s,desired_rate,desired_sample_width)
        
    #Finally, convert/trim also the existing .wav files that were already there before (no suffix 'wav_suffix') 

    for wav_file in wav_list:

        if not is_converted_wav(wav_file,suffix=wav_suffix):
            
            tempaudio = AudioSegment.from_file(folder+wav_file,'wav')
                          
            wav_filename_no_extension = wav_file.split(sep='.wav')[0]
            
            wav_filename_w_suffix = wav_filename_no_extension + wav_suffix + '.wav'
            
            export_AudioSegment_to_wav(tempaudio,folder+wav_filename_w_suffix,trim_loc=trim_loc,desired_s=desired_s,desired_rate=desired_rate,desired_sample_width=desired_sample_width)
 
#Another helper function that returns the original filename (before the suffix was added)    
def filename_before_trim(trimmed_wav_file,suffix='_conv',max_len=20):
    """
    Returns the filename of a trimmed file (i.e. the name before the suffix was added), possibly trimmed to max_len characters at most
    """  
    return trimmed_wav_file.split(sep=(suffix+'.wav'))[0][:max_len]


## Create a function that returns the mean (time average) of a bunch of features of the audio (a subset of the features returned in FMA)

def feats_from_audio(wav_audiofile,srate=22050):
    
    x, _ = librosa.load(wav_audiofile,sr=srate)

    feats = dict()
    
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=srate)[0]
    feats['spectral_centroids_mean'] = np.mean(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=srate)[0] #(x+0.01, sr=srate)[0]
    feats['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x, sr=srate)[0] #(x+0.01, sr=srate)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x, sr=srate, p=3)[0] #(x+0.01, sr=srate, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x, sr=srate, p=4)[0] #(x+0.01, sr=srate, p=4)[0]
    feats['spectral_bandwidth_2_mean'] = np.mean(spectral_bandwidth_2)
    feats['spectral_bandwidth_3_mean'] = np.mean(spectral_bandwidth_3)
    feats['spectral_bandwidth_4_mean'] = np.mean(spectral_bandwidth_4)
    
    zcr = librosa.feature.zero_crossing_rate(x)
    feats['zcr_mean'] = np.mean(zcr)
    feats['zcr_max'] = np.max(zcr)
    
    rootmeansquare = librosa.feature.rms(x)
    feats['rms_mean'] = np.mean(rootmeansquare)
    
    chromagram = librosa.feature.chroma_stft(x, sr=srate)
    feats['Chroma_means'] = np.mean(chromagram,axis=1)
    
    mfccs = librosa.feature.mfcc(x, sr=srate)
    feats['MFCCs_means'] = np.mean(mfccs,axis=1)
    
    return feats
