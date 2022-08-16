#!/usr/bin/env python3

"""This module creates BaselineModels for the Spotify MPD Playlist Challenge."""

# ---------- Imports ---------- #
import random
import os
import csv

from tqdm import tqdm
tqdm.pandas()

import numpy as np
import pandas as pd

import tensorflow as tf


# ---------- BaselineModels Class ---------- #

class BaselineModels:
    """This class offers naive baseline recommendation behavior to compare
    machine learning based recommendations against."""

    def __init__(self, playlist_df, track_df):

        # Save arguments
        self.playlist_df = playlist_df
        self.track_df = track_df
        self.playlist_dict, self.track_dict = self.playlist_df.to_dict(), self.track_df.to_dict()


    def get_track_feature(self, uri, feature='track_name'):
        """
        Given a track uri (as a string) and the track dictionary, return the requested feature.
        
        Features include:
            - 'track_name'
            - 'album_name'
            - 'album_uri'
            - 'artist_name'
            - 'artist_uri'
            - 'duration_ms'
        """
        try:
            return self.track_df[feature][uri]
        except Exception as e:
            return e


    def calculate_track_popularities(self, pop_col_name='popularity'):
        """Create counts of song appearances in playlists and save in a new column with value of <pop_col> parameter."""

        # Local copy to work with
        track_df = self.track_df

        # Columns in playlist_df with track URIs
        track_uri_cols = [col for col in self.playlist_df.columns
                            if 'uri' in col
                            and 'album' not in col
                            and 'artist' not in col]

        # Use dictionary to hold song appearances
        track_counts_dict = {}
        for col in track_uri_cols:
            for uri in self.playlist_df[col].tolist():
                if uri not in track_counts_dict.keys():
                    track_counts_dict.update({uri: 1})
                else:
                    track_counts_dict.update({uri: track_counts_dict.get(uri) + 1})

        # Add dictionary values to track_df
        track_df.reset_index(inplace=True)
        track_df['appears_in_count'] = track_df.reset_index(drop=True).apply(lambda row: int(track_counts_dict.get(row.track_uri, 0)), axis=1)
        track_df.set_index('track_uri', inplace=True)

        # Update self.track_df
        self.track_df = track_df


    def recommend_tracks_by_artists(self, num_tracks=5, input_track_uris=[], popularity=True):
        """
        Recommend tracks by the same artists as those in input_track_uris.
        If `popularity` is set to True, recommend popular songs by the same artist;
        if False, recommend random songs by the same artist.
        """
        
        recommended_track_uris = []

        # Create list of artists presented in input tracks
        input_artists = []
        for track_uri in input_track_uris:
            artist_uri = self.get_track_feature(track_uri, 'artist_uri')
            input_artists.append(artist_uri)

        # RECOMMENDING SONGS BY POPULARITY
        if popularity:
            # Create recommendations by randomly picking an artist
            # and finding their most popular tracks
            # that aren't already in the playlist
            while len(recommended_track_uris) < num_tracks:

                # Pick a random artist to get songs by
                this_artist = random.choice(input_artists)

                # Find this artist's top songs and save track_uris in a list
                artist_top_songs = self.track_df.sort_values(by='popularity', ascending=False)
                artist_top_songs = artist_top_songs[artist_top_songs.artist_uri == this_artist]\
                                    .reset_index()\
                                    ['track_uri']\
                                    .to_list()

                # If this artist only has one song, pick a different artist from the list
                if len(artist_top_songs) == 1:
                    continue

                # Loop through in order of popularity and add songs not already in playlist
                # and not already in recommendation list
                while True:

                    # Try to pop, if fails then break and pick new artist
                    try:
                        try_song = artist_top_songs.pop(0)
                    except Exception as e:
                        break

                    if try_song in input_track_uris or try_song in recommended_track_uris:
                        # If this song is already in the playlist, do nothing
                        # If there are no more songs left to try to recommend, break without appending
                        if len(artist_top_songs) == 0:
                            break
                    else:
                        # If not already in playlist, add this song to the recommendations and break
                        recommended_track_uris.append(try_song)
                        break
        
        # RECOMMENDING ARTIST SONGS RANDOMLY
        else:
            # Create recommendations by randomly picking an artist
            # and recommending a random song by them
            while len(recommended_track_uris) < num_tracks:

                # Pick a random artist to get songs by
                this_artist = random.choice(input_artists)

                # Find this artist's songs and save track_uris in a list
                # If no results found, repeat loop
                while True:
                    try:
                        artist_songs_df = self.track_df[self.track_df.artist_uri == this_artist].reset_index()
                        break
                    except Exception as e:
                        continue

                # If this artist only has one song, pick a different artist from the list
                if artist_songs_df.shape[0] == 1:
                    continue

                # Loop until we find a song not already in the playlist
                attempts = 0
                while True and attempts <= artist_songs_df.shape[0]:
                    try_song = artist_songs_df.sample()['track_uri'].iloc[0]

                    if try_song in input_track_uris or try_song in recommended_track_uris:
                        # If this song is already in the playlist, do nothing.
                        attempts += 1
                        pass
                    else:
                        # If not already in playlist, add this song to the recommendations and break
                        recommended_track_uris.append(try_song)
                        break

        # Return recommendations
        return recommended_track_uris


    def recommend_tracks_by_albums(self, num_tracks=5, input_track_uris=[], popularity=True):
        """
        Recommend tracks from the same albums as those in input_track_uris.
        If `popularity` is set to True, recommend popular songs from the same album;
        if False, recommend random songs from the same album.
        """

        recommended_track_uris = []

        # Create list of albums presented in input tracks
        input_albums = []
        for track_uri in input_track_uris:
            album_uri = self.get_track_feature(track_uri, 'album_uri')
            input_albums.append(album_uri)

        # RECOMMENDING SONGS BY POPULARITY
        if popularity:
            # Create recommendations by randomly picking an album
            # and finding its most popular tracks
            # that aren't already in the playlist
            while len(recommended_track_uris) < num_tracks:

                # Pick a random album to get songs from
                this_album = random.choice(input_albums)

                # Find this album's top songs and save track_uris in a list
                album_top_songs = self.track_df.sort_values(by='popularity', ascending=False)
                album_top_songs = album_top_songs[album_top_songs.album_uri == this_album]\
                                    .reset_index()\
                                    ['track_uri']\
                                    .to_list()

                # If this artist only has one song, pick a different album from the list
                # if len(album_top_songs) == 1:
                #     continue

                # Loop through in order of popularity and add songs not already in playlist
                while True:

                     # Try to pop, if fails then break and pick new artist
                    try:
                        try_song = album_top_songs.pop(0)
                    except Exception as e:
                        break
                    
                    if len(album_top_songs) == 1 and try_song in recommended_track_uris:
                        # If there is only one song on the album and it's already in the
                        # recommended songs, just add it again
                        recommended_track_uris.append(try_song)
                        break

                    if try_song in input_track_uris or try_song in recommended_track_uris:
                        # If this song is already in the playlist, do nothing
                        # If there are no more songs left to try to recommend, break without appending
                        if len(album_top_songs) == 0:
                            break
                    else:
                        # If not already in playlist, add this song to the recommendations and break
                        recommended_track_uris.append(try_song)
                        break
        
        # RECOMMENDING ALBUM SONGS RANDOMLY
        else:
            # Create recommendations by randomly picking an album
            # and recommending a random song from it
            while len(recommended_track_uris) < num_tracks:

                # Pick a random artist to get songs by
                this_album = random.choice(input_albums)

                # Find this album's songs and save track_uris in a list
                # If no results found, repeat loop
                while True:
                    try:
                        album_songs_df = self.track_df[self.track_df.album_uri == this_album].reset_index()
                        break
                    except Exception as e:
                        continue

                # If this album only has one song, pick a different album from the list
                if album_songs_df.shape[0] == 1:
                    continue

                # Loop until we find a song not already in the playlist
                attempts = 0
                while True and attempts <= album_songs_df.shape[0]:
                    try:
                        try_song = album_songs_df.sample()['track_uri'].iloc[0]
                    except Exception as e:
                        break

                    if try_song in input_track_uris or try_song in recommended_track_uris:
                        # If this song is already in the playlist, do nothing.
                        attempts += 1
                        pass
                    else:
                        # If not already in playlist, add this song to the recommendations and break
                        recommended_track_uris.append(try_song)
                        break

        # Return recommendations
        return recommended_track_uris
                

    def recommend_popular_tracks(self, num_tracks=5, input_track_uris=[], popularity=True):
        """
        Recommend the most popular tracks in our dataset.
        """

        recommended_track_uris = []
        
        while len(recommended_track_uris) < num_tracks:

            # Order all songs by popularity and save track_uris to a list
            pop_songs = self.track_df.\
                sort_values(by='popularity', ascending=False)\
                .reset_index()['track_uri']\
                .to_list()

            # Loop through in order of popularity and add songs not already in playlist
            while True:
                try_song = pop_songs.pop(0)
                if try_song in input_track_uris or try_song in recommended_track_uris:
                    # If this song is already in the playlist, do nothing
                    pass
                else:
                    # If not already in playlist, add this song to the recommendations and break
                    recommended_track_uris.append(try_song)
                    break

        # Return recommendations
        return recommended_track_uris
