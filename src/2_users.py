import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import utils

def get_users(
    meta_path: str,
    label_path: str,
    user_path: str,
    subreddit: str,
    n_clusters: int,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    q: float,
):
    t0 = time.time()
    years = range(start_year, end_year+1)
    months = range(start_month, end_month+1)
    
    print(f'CPU count  : {os.cpu_count()}')
    print(f'Subreddit  : {subreddit}')
    print(f'Range      : {start_year}-{start_month} to {end_year}-{end_month}\n')

    all_top_users = []

    # pull all metadata for subreddit and get top q-th users
    for year, month in [(yr, mo) for yr in years for mo in months]:
        print(f'Reading {year}-{month:02} ... ({time.time()-t0:.3f})')
        
        # load metadata for this subreddit
        print(f'> Loading metadata ... ({time.time()-t0:.3f})')
        metadata = pd.read_csv(
            os.path.join(meta_path, f'metadata_{year}-{month:02}.csv'),
            compression='gzip',
        )
        metadata = metadata[metadata['subreddit'] == subreddit]

        # load labels
        with open(os.path.join(label_path, f'labels_{year}-{month:02}.npz'), 'rb') as f:
            labels = np.load(f)['labels']

        # Add labels as a column of metadata
        metadata['label'] = labels

        # Keep only the top users based on counts
        user_counts = metadata['author'].value_counts()
        top_users = user_counts[user_counts >= user_counts.quantile(1 - q)]
        all_top_users.append(top_users)

        # Filter metadata to include only top users
        top_users_metadata = metadata[metadata['author'].isin(top_users.index)]

        # Create a pivot table with users as rows and labels as columns
        user_label_counts = top_users_metadata.pivot_table(
            index='author', 
            columns='label', 
            aggfunc='size', 
            fill_value=0
        )

        # Save the user-label counts to a CSV file
        user_label_counts.to_csv(
            os.path.join(user_path, f'user_label_counts_{year}-{month:02}.csv')
        )

    # concatenate into a single series
    all_top_users = pd.concat(all_top_users)
    all_top_users = all_top_users.groupby(all_top_users.index).sum()


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('../config.ini')
    g = config['general']

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpath', type=str, required=True)
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--start-month', type=int, required=True)
    parser.add_argument('--end-month', type=int, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--q', type=float, default=0.1, required=True)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)
    
    subdir = 'users'
    if not os.path.exists(os.path.join(subpath, subdir)):
        os.makedirs(os.path.join(subpath, subdir))
    
    get_users(
        meta_path=g['meta_path'],
        label_path=os.path.join(subpath, 'labels'),
        user_path=os.path.join(subpath, subdir),
        subreddit=args.subreddit,
        n_clusters=args.n_clusters,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        quartile=args.q,
    )