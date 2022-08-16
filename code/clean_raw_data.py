"""
This script parse's Spotify's MPD Dataset.

Usage:
    python3 clean_raw_data.py <path-to-data-directory>

The JSON slice files are converted to CSV slices. The result is:

- playlists.csv
- tracks.csv

"""


# Imports
import numpy as np
import pandas as pd
import sys
import os
import json
import csv


def process_mpd_data(path='', num_tracks=15):

    # --- Use default input path if one isn't provided
    path = "../spotify_million_playlist_dataset/data/"
    mpd_filepath = os.path.relpath(path)
    print('Input directory:')
    print(mpd_filepath)

    # --- Define output paths
    # Base output directory
    out_path = '../data'
    output_filepath_root = os.path.relpath(out_path)

    # Output playlists into playlists directory
    playlist_output = os.path.join(output_filepath_root, 'playlists')

    # Output tracks into tracks directory
    track_output = os.path.join(output_filepath_root, 'tracks')

    print('Output directories:')
    print(output_filepath_root)
    print(playlist_output)
    print(track_output)

    # --- 
    # Input for number of tracks to include as features for each playlist
    num_tracks = num_tracks

    # Begin parsing JSON files
    print('\nParsing JSONs...\n')
    length = len(os.listdir(mpd_filepath))

    for ii, ff in enumerate(os.listdir(mpd_filepath)):
        print(f"Slice {ii + 1} of {length}: {ff}")
        suffix = str(ff.split('.')[-2])

        # Call parsing helper function
        parse_spotify_mpd_json_to_csv(
            input_filepath=os.path.join(mpd_filepath, ff),
            output_filepaths=[playlist_output, track_output],
            playlist_csv_name='mpd_playlists_' + suffix,
            tracks_csv_name='mpd_tracks_' + suffix,
            tracks_per_playlist=num_tracks)

        print()




def parse_spotify_mpd_json_to_csv(input_filepath, output_filepaths, playlist_csv_name, tracks_csv_name, tracks_per_playlist=15):
    """Parse through Spotify's Million Playlist Dataset (MPD) JSON slice file and produce the following CSV files:
    
        - {playlist_csv_name}.csv
        - {tracks_csv_name}.csv
        
        * Note: '.csv' suffice not required in argument string.
        
        
    The MPD JSON has a *PLAYLIST* field for each playlist row. The *INFO* field is ignored for each slice file.
    
    The JSON is structured in the following format:
    
        ### `info` Field (THE INFO FIELD IS IGNORED)
        

        ### `playlists` field 
        This is an array that typically contains 1,000 playlists. Each playlist is a dictionary that contains the following fields:

            * ***pid*** - integer - playlist id - the MPD ID of this playlist.
            * ***name*** - string - the name of the playlist 
            * ***description*** - optional string - if present, the description given to the playlist.
            * ***modified_at*** - seconds - timestamp (in seconds since the epoch) when this playlist was last updated.
            * ***num_artists*** - the total number of unique artists for the tracks in the playlist.
            * ***num_albums*** - the number of unique albums for the tracks in the playlist
            * ***num_tracks*** - the number of tracks in the playlist
            * ***num_followers*** - the number of followers this playlist had at the time the MPD was created.
            * ***num_edits*** - the number of separate editing sessions.
            * ***duration_ms*** - the total duration of all the tracks in the playlist (in milliseconds)
            * ***collaborative*** -  boolean - if true, the playlist is a collaborative playlist. 
            * ***tracks*** - an array of information about each track in the playlist. Each element is a dictionary with:
               * ***track_name*** - the name of the track
               * ***track_uri*** - the Spotify URI of the track
               * ***album_name*** - the name of the track's album
               * ***album_uri*** - the Spotify URI of the album
               * ***artist_name*** - the name of the track's primary artist
               * ***artist_uri*** - the Spotify URI of track's primary artist
               * ***duration_ms*** - the duration of the track in milliseconds
               * ***pos*** - the position of the track in the playlist (zero-based)
        
    
    This function does not further process the rows, it simply converts the JSON to CSVs.
    Rows are not sorted, duplicates are not removed.
    
    """
    
    # Load JSON from provided filepath
    j = ''
    with open(input_filepath, 'r') as file:
        j = json.load(file)
        
    # Lists to gather from nested JSON
    playlists_json_list = []
    tracks_json_list = []
    
    # MPD JSON structure has PLAYLISTs as outermost dictionary
    # Step through PLAYLISTs
    for playlist in j['playlists']:
        
        # -- Create this playlist's playlist row as JSON for CSV creation
        playlist_json = {}
        playlist_json['pid'] = playlist['pid']
        playlist_json['name'] = playlist['name']

        # Description is an optional field
        if 'description' in playlist:
            playlist_json['description'] = playlist['description']
        else:
            playlist_json['description'] = ''
            
        playlist_json['modified_at'] = playlist['modified_at']
        playlist_json['num_artists'] = playlist['num_artists']
        playlist_json['num_albums'] = playlist['num_albums']
        playlist_json['num_tracks'] = playlist['num_tracks']
        playlist_json['num_followers'] = playlist['num_followers']
        playlist_json['num_edits'] = playlist['num_edits']
        playlist_json['duration_ms'] = playlist['duration_ms']
        playlist_json['collaborative'] = playlist['collaborative']
        
        playlists_json_list.append(playlist_json)
        
        
        # -- Creating tracks csv
        # Each playlist has any number of tracks.
        # Use the *PID* to affiliate tracks with playlists
        
        # -- Saving tracks for each playlist
        # Add n tracks as features to the playlist csv
        # Where n is the input tracks_per_playlist
        # Format: track_1_uri, track_2_uri,...
        #         track_1_album_uri
        #         track_1_artist_uri
        #
        # Use padding where track does not exist (only 5 tracks in playlist when we want 15)

        track_counter = 1
        
        for track in playlist['tracks']:
            
            if track_counter <= tracks_per_playlist:
                
                # Track URI
                track_feature_name = 'track_' + str(track_counter) + '_uri'
                playlist_json[track_feature_name] = track['track_uri']
                
                # Album URI
                album_feature_name = 'track_' + str(track_counter) + '_album_uri'
                playlist_json[album_feature_name] = track['album_uri']
                
                # Arist URI
                artist_feature_name = 'track_' + str(track_counter) + '_artist_uri'
                playlist_json[artist_feature_name] = track['artist_uri']
            
            track_counter += 1

            # Create tracks CSV
            tracks_json = {}
            tracks_json['pid'] = playlist['pid']
            tracks_json['track_name'] = track['track_name']
            tracks_json['track_uri'] = track['track_uri']
            tracks_json['album_name'] = track['album_name']
            tracks_json['album_uri'] = track['album_uri']
            tracks_json['artist_name'] = track['artist_name']
            tracks_json['artist_uri'] = track['artist_uri']
            tracks_json['duration_ms'] = track['duration_ms']
            tracks_json['pos'] = track['pos']
            
            tracks_json_list.append(tracks_json)
            
        # Check if there weren't enough tracks and we need to add padding
        while track_counter <= tracks_per_playlist:
            # Resume track_counter, but now we assign '0' values
            # Track URI
            track_feature_name = 'track_' + str(track_counter) + '_uri'
            playlist_json[track_feature_name] = 0

            # Album URI
            album_feature_name = 'track_' + str(track_counter) + '_album_uri'
            playlist_json[album_feature_name] = 0

            # Arist URI
            artist_feature_name = 'track_' + str(track_counter) + '_artist_uri'
            playlist_json[artist_feature_name] = 0
            
            track_counter += 1
            
    # Now that we've parsed through the JSON, save data to CSV
    
    # mpd_playlists.csv
    with open(f"{output_filepaths[0]}/{playlist_csv_name}.csv", 'w') as file:
        dw = csv.DictWriter(file, playlists_json_list[0].keys())
        dw.writeheader()
        dw.writerows(playlists_json_list)
        print(f"{playlist_csv_name}.csv saved to {output_filepaths[0]}")
    
    # mpd_tracks.csv
    with open(f"{output_filepaths[1]}/{tracks_csv_name}.csv", 'w') as file:
        dw = csv.DictWriter(file, tracks_json_list[0].keys())
        dw.writeheader()
        dw.writerows(tracks_json_list)
        print(f"{tracks_csv_name}.csv saved to {output_filepaths[1]}")
        

def delete_data_files(playlist_output, track_output):
    """
    Prompts user to delete previously-produced output files in playlsit and track directories.
    """

    if len(os.listdir(playlist_output)) > 0 or len(os.listdir(track_output)) > 0:
        if input("Files detected in output directories. Shall I delete them for you? (Y/N) ").lower() == 'y':
            for file in os.listdir(playlist_output):
                path = os.path.join(playlist_output, file)
                
                try:
                    os.remove(path)
                except OSError as e:
                    print("Error: %s : %s" % (file, e.strerror))
            print(f'Playlist data files deleted from {playlist_output}.')
                
            for file in os.listdir(track_output):
                path = os.path.join(track_output, file)
                
                try:
                    os.remove(path)
                except OSError as e:
                    print("Error: %s : %s" % (path, e.strerror))
                    
            print(f'Track data files deleted from {track_output}.')

            # Files deleted - return proceed code
            return 1
            
        else:
            # If user does not delete, return stop code
            print("No files were deleted.")
            return 0
    else:
        # If no files detected, return proceed code
        print("No files detected in output folders. You're ready to parse the raw data.")
        return 1


# Running from Command Line
if __name__ == "__main__":
    path = sys.argv[1]
    num_tracks = sys.argv[2]
    # if len(sys.argv) > 2 and sys.argv[2] == "--quick":
    #     quick = True
    # process_mpd(path)