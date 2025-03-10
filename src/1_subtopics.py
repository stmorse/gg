import argparse
import configparser
import gc
import heapq        # for top-k cluster members
import json
import os
import pickle
import time

def find_topics():
    """
    1. load metadata of subreddit
    2. load embeddings of subreddit
    3. cluster embeddings
    4. save labels
    5. save top-k words
    6. compute tf-idf
    """  

    # load metadata
    print('Loading metadata ... ', end=' ')
    t0 = time.time()

    return



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
    parser.add_argument('--top-k', type=int, default=100, required=False)
    parser.add_argument('--top-m', type=int, default=20, required=False)
    parser.add_argument('--max-df', type=float, default=0.3, required=False)
    args = parser.parse_args()

    subpath = os.path.join(g['save_path'], args.subpath)
    
    for subdir in ['labels', 'models', 'tfidf']:
        if not os.path.exists(os.path.join(subpath, subdir)):
            os.makedirs(os.path.join(subpath, subdir))
    
    find_topics(
        data_path=g['data_path'],
        embed_path=g['embed_path'],
        label_path=os.path.join(subpath, 'labels'),
        model_path=os.path.join(subpath, 'models'),
        tfidf_path=os.path.join(subpath, 'tfidf'),
        n_clusters=args.n_clusters,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        top_k=args.top_k,
        top_m=args.top_m,
        max_df=args.max_df
    )