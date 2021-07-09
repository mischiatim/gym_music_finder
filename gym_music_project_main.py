# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 23:35:40 2021

A fun project to compute features from music in my library,
classify new songs as Gym/NoGym based on some labeled songs in my library,
and find music in the Free Music Archive (which I downloaded 
separately) with similar features to any one song in my library

@author: matteo mischiati

"""

#module with my audio processing helper functions
import audio_processing as ap

# import the (slightly edited) feature extraction code from the Free Music Archive (FMA)
import features_FMA

# general imports
import pandas as pd
import numpy as np

#I want to trim and convert all music in my library (.mp3, .aac, .wma) to .wav files with the same duration,
#sampling rate and sample width for further processing. I also add a suffix to detect songs already 
#trimmed and ready for further processing
srate = 22050
swidth = 2
trim_duration_s = 30
trim_suffix = '_trim'

#All labeled songs (that I have already categorized as good or bad for the gym) are in two separate
#subfolders 'Gym_music_library' and 'NoGym_music_library'
gym_folder = './Gym_Music_library/'
nogym_folder = './NoGym_music_library/'

ap.convert_subfolder_audios_to_wav(folder=gym_folder,wav_suffix=trim_suffix,trim_loc='mid',desired_s=trim_duration_s,desired_rate=srate,desired_sample_width=swidth)
ap.convert_subfolder_audios_to_wav(folder=nogym_folder,wav_suffix=trim_suffix,trim_loc='mid',desired_s=trim_duration_s,desired_rate=srate,desired_sample_width=swidth)

#Also trim and convert the folder of all songs that I haven't classified yet, and would like to 
#assess whether they are good or bad for a gym playlist 
newsongs_folder='./Music_library_to_test/'
ap.convert_subfolder_audios_to_wav(folder=newsongs_folder,wav_suffix=trim_suffix,trim_loc='mid',desired_s=trim_duration_s,desired_rate=srate,desired_sample_width=swidth)

#Create dataframe of features for all the labeled gym songs (from the trimmed .wav files):
wav_list_gym = ap.filetype_search('wav',gym_folder)
converted_wav_list_gym =  list(filter(lambda wav_file: ap.is_converted_wav(wav_file,trim_suffix),wav_list_gym))
print('\nCreating dataframe of features for all the labeled gym songs:\n')
print(converted_wav_list_gym)

def filename_before_trim(trimmed_wav_file,suffix='_conv',max_len=20):
    """
    Returns the filename of a trimmed file (i.e. the name before the suffix was added), possibly trimmed to max_len characters at most
    """  
    return trimmed_wav_file.split(sep=(suffix+'.wav'))[0][:max_len]

max_len = 20
converted_wav_list_gym_names = [filename_before_trim(wav_file,trim_suffix,max_len) for wav_file in converted_wav_list_gym]

gym_songs_FMAfeats_df = pd.DataFrame(index=converted_wav_list_gym_names,columns=features_FMA.columns(), dtype=np.float32)

for idx, gym_song in enumerate(converted_wav_list_gym): 
    gym_songs_FMAfeats_df.iloc[idx] = features_FMA.compute_features_from_filename(gym_folder+gym_song,srate=srate)

#add a column with the Target class (is_gym)
gym_songs_FMAfeats_df['is_gym_song']=True

print(gym_songs_FMAfeats_df)

#Create similar dataframe of features as above but for all the labeled No-gym songs (from the trimmed .wav files):

wav_list_nogym = ap.filetype_search('wav',nogym_folder)
converted_wav_list_nogym =  list(filter(lambda wav_file: ap.is_converted_wav(wav_file,trim_suffix),wav_list_nogym))
print('\nCreating dataframe of features for all the labeled No-gym songs:\n')
print(converted_wav_list_nogym)

converted_wav_list_nogym_names = [filename_before_trim(wav_file,trim_suffix,max_len) for wav_file in converted_wav_list_nogym]

nogym_songs_FMAfeats_df = pd.DataFrame(index=converted_wav_list_nogym_names,columns=features_FMA.columns(), dtype=np.float32)

for idx, nogym_song in enumerate(converted_wav_list_nogym): 
    nogym_songs_FMAfeats_df.iloc[idx] = features_FMA.compute_features_from_filename(nogym_folder+nogym_song,srate=srate)

#add a column with the Target class (is_gym)
nogym_songs_FMAfeats_df['is_gym_song']=False

print(nogym_songs_FMAfeats_df)

#Save the two dataframes of features (for labeled Gym and No-gym songs) to .csv format:
nogym_songs_FMAfeats_df.to_csv('Nogym_songs_FMAfeatures.csv')
gym_songs_FMAfeats_df.to_csv('Gym_songs_FMAfeatures.csv')

#Create similar dataframe of features as above but for all the new songs to be classified as Gym vs No-gym (from the trimmed .wav files):
#(of course this dataframe will not have the target class)

wav_list_new = ap.filetype_search('wav',newsongs_folder)
converted_wav_list_new =  list(filter(lambda wav_file: ap.is_converted_wav(wav_file,trim_suffix),wav_list_new))
print('\nCreating dataframe of features for all the songs to be labeled:\n')
print(converted_wav_list_new)

converted_wav_list_new_names = [filename_before_trim(wav_file,trim_suffix,max_len) for wav_file in converted_wav_list_new]

new_songs_FMAfeats_df = pd.DataFrame(index=converted_wav_list_new_names,columns=features_FMA.columns(), dtype=np.float32)

for idx, new_song in enumerate(converted_wav_list_new): 
    new_songs_FMAfeats_df.iloc[idx] = features_FMA.compute_features_from_filename(newsongs_folder+new_song,srate=srate)


#Combine the Gym and No-Gym labeled dtaframes and use them to train a classifier 
X_nogym = nogym_songs_FMAfeats_df.drop('is_gym_song',axis=1)
y_nogym = nogym_songs_FMAfeats_df['is_gym_song']

X_gym = gym_songs_FMAfeats_df.drop('is_gym_song',axis=1)
y_gym = gym_songs_FMAfeats_df['is_gym_song']

X_train = X_gym.append(X_nogym)
y_train = y_gym.append(y_nogym)

# I use a Nearest Neighbor classifier as I want to have interpretable results to see
#if the model makes predictions that make sense

#I need to scale the features otherwise the distance metric may be dominated by a few of them
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
#Need to then also apply the same scaling to the features of the songs to be tested
X_new_scaled = scaler.transform(new_songs_FMAfeats_df)

#Need to also apply PCA reduction (or another dimensionality reduction technique)
#or the solution will suffer from the course of dimensionality 
#(because the feature space is very high-dimensional distances become very noisy)
#Also, we may have fewer training samples (number of labeled songs) than features (>500)
#which would lead to overfitting

from sklearn.decomposition import PCA

#The best option is to compute PCA components on a larger dataset, 
#such as all the features of the Free Music Archive 
FMAarchive_feats_df=pd.read_csv('./FreeMusicArchive_metadata/features.csv',header=[0,1,2],skiprows=[3],index_col=[0])

#Subselect the tracks that are in the 'small FMA dataset', that were 8000 song snippets (30s each) from 8 different genres
#For that, I need to look at the 'tracks.csv' datafile:
FMAarchive_tracks_df=pd.read_csv('./FreeMusicArchive_metadata/tracks.csv',header=[0,1],index_col=[0])    
FMAarchive_tracks_df_small = FMAarchive_tracks_df[FMAarchive_tracks_df['set','subset']=='small']
FMAarchive_feats_df_small = FMAarchive_feats_df[FMAarchive_tracks_df['set','subset']=='small']

X_FMA_scaled = scaler.transform(FMAarchive_feats_df_small)

#Note: I needed to subsample the tracks or the matrix on which to perform pca would be too large to handle
pca = PCA(n_components=30)
pca.fit(X_FMA_scaled[0::5,:])

## This would be the alternative method if we didn't have the FMA 
## archive and we only used the current training set of songs
#pca = PCA(n_components=X_train_scaled.shape[0])
#pca.fit(X_train_scaled)

#Optionally, visualize the amount of variance explained by each PCA dimension to choose how many to keep
import matplotlib.pyplot as plt
plt.figure(figsize=[12,6])
plt.subplot(1,2,1)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('component')
plt.ylabel('explained variance')
plt.subplot(1,2,2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#Keep only components that explain at least 1% of the variance
n_comp_pca_chosen = np.where(pca.explained_variance_ratio_>0.01)[0][-1]

pca = PCA(n_components=n_comp_pca_chosen)
pca.fit(X_train_scaled)

X_train_scaled_pca = pca.transform(X_train_scaled)
#Need to then also apply the same reduction to the features of the songs to be tested
X_new_scaled_pca = pca.transform(X_new_scaled)

#And since I have it, I apply the PCA transformation to the FMA songs as well
X_FMA_scaled_pca = pca.transform(X_FMA_scaled)

#First I try a Nearest Neighbor classifier to have something whose outputs can be assessed
#easily: does the closest song (which is used to classify a song as Gym/NoGym) 
#to a given song make sense?
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='cosine') #,metric='euclidean') #metric='cityblock',p=2 #metric='minkowski',p=2
knn.fit(X_train_scaled_pca,y_train)

predictions_new_songs_KNN_1 = knn.predict(X_new_scaled_pca)

nearest_neighbors_list_new_songs = knn.kneighbors(X_new_scaled_pca)[1]

for i,song in enumerate(new_songs_FMAfeats_df.index):
    print('\n Closest song to {} is {}\n'.format(song,X_train.index[nearest_neighbors_list_new_songs[i][0]]))
    
#Optionally, let's actually plot the distances between songs in the training set to see if they
#make sense:

import seaborn as sns
from scipy.spatial.distance import pdist, squareform, cdist

plt.figure(figsize=[16,16])
sns.heatmap(squareform(pdist(X_train_scaled_pca,metric='cosine')),xticklabels=X_train.index,yticklabels=X_train.index)

#There seem to be 3-4 clusters, let's try to find them

#How many clusters do we expect? Let's plot the first 2 PCA components of the features and see 
#(but note that this visualizes the euclidean distance)
plt.figure(figsize=[6,6])
plt.scatter(X_train_scaled_pca[:,0],X_train_scaled_pca[:,1])

#Let's see if tSNE helps to better visualize the data
from sklearn.manifold import TSNE
X_train_scaled_pca_embedded = TSNE(n_components=2).fit_transform(X_train_scaled_pca)
X_train_scaled_pca_embedded.shape

plt.figure(figsize=[6,6])
plt.scatter(X_train_scaled_pca_embedded[:,0],X_train_scaled_pca_embedded[:,1])

#It doesn't look like the visualization helps... 
#let's try with affinity propagation, that tries to estimate number of clusters

from sklearn.cluster import AffinityPropagation

# Compute Affinity Propagation
af = AffinityPropagation().fit(X_train_scaled_pca)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

cluster_labels = af.predict(X_train_scaled_pca)

ordered_song_nums = np.argsort(cluster_labels)

X_train_scaled_pca_reorder= X_train_scaled_pca[ordered_song_nums,:]
X_train_scaled_pca_index_reorder=X_train.index[ordered_song_nums]

plt.figure(figsize=[16,16])
sns.heatmap(squareform(pdist(X_train_scaled_pca_reorder,metric='cosine')),xticklabels=X_train_scaled_pca_index_reorder,yticklabels=X_train_scaled_pca_index_reorder)

#Now create code to find the distance from any one given song of every other song from a full library 
#(ordering songs from closest to furthest in parameter space)

def return_closest_songs(seed_song_features,song_library_features,song_library_names,dist_metric='cosine'):
    vector_distances = cdist(seed_song_features,song_library_features,metric=dist_metric)

    ordered_vector_distances_idx = np.argsort(vector_distances)
    
    ordered_vector_distances = vector_distances[0][ordered_vector_distances_idx]
    
    song_library_names_reordered = song_library_names[ordered_vector_distances_idx]
    
    return song_library_names_reordered[0], ordered_vector_distances[0]
    
#That's how the function can be used to order the songs in X_train based on distance to 'Costruire':
seed_song_title = 'Costruire'
index_song =  np.where(X_train.index==seed_song_title)
seed_song_features = X_train_scaled_pca[index_song[0],:]

X_train_songs_closest_to_seed, dist_X_train_songs_closest_to_seed = return_closest_songs(seed_song_features,X_train_scaled_pca,X_train.index,dist_metric='cosine')

print(X_train_songs_closest_to_seed)
print(dist_X_train_songs_closest_to_seed)

#This is how the function can be used to cross check closest song in training set for each of the new songs
for i,song in enumerate(new_songs_FMAfeats_df.index):
    X_train_songs_closest_to_seed, dist_X_train_songs_closest_to_seed  = return_closest_songs(X_new_scaled_pca[[i],:],X_train_scaled_pca,X_train.index,dist_metric='cosine')
    print('\n Closest song to {} is {} (dist={})\n'.format(song,X_train_songs_closest_to_seed[0],dist_X_train_songs_closest_to_seed[0]))
    
#And finally here is the most useful function: given a song in my library (e.g. in X_train), 
#find closest songs within the Free Music Archive (FMA_small) dataset
seed_song_title = '02 Pavarotti Traccia' 
index_song =  np.where(X_train.index==seed_song_title)
seed_song_features_scaled_pca = X_train_scaled_pca[index_song[0],:]
seed_song_features_scaled_pca.shape

FMAarchive_tracks_df_small_track_titles =[FMAarchive_tracks_df_small[FMAarchive_tracks_df_small.index == i]['track','title'].values[0] for i in FMAarchive_tracks_df_small.index]
closest_trackids_in_FMA , closest_distances_in_FMA = return_closest_songs(seed_song_features_scaled_pca,X_FMA_scaled_pca,FMAarchive_tracks_df_small.index,dist_metric='cosine')
#print(closest_distances_in_FMA)
print(closest_trackids_in_FMA)
closest_tracktitles_in_FMA =[FMAarchive_tracks_df_small[FMAarchive_tracks_df_small.index == i]['track','title'].values[0] for i in closest_trackids_in_FMA]
print(closest_tracktitles_in_FMA[0:10])