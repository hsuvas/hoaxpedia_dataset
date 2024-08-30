"""
Function Call: python collect_real.py --data_path ../data/hoax_unified_v4.csv --output_path ../data

"""

import pandas as pd
from utils import get_revision_timestamps, clean_wiki_article, remove_old_revision, remove_current_revision, add_space_after_full_stop
import argparse
import requests

if __name__ == '__main__':
    #set up the argument parser
    parser = argparse.ArgumentParser(description='Collect real data')
    parser.add_argument('--data_path', type=str, default='../data/hoax_unified_v4.csv', help='Path to the hoax data file')
    parser.add_argument('--output_path', type=str, default='.', help='Path to save the real data file')
    args = parser.parse_args()

    input_path = args.data_path
    output_path = args.output_path

    #load the hoax data
    df_top_titles = pd.read_csv(input_path)
    df_top_titles.top_100_NNs = df_top_titles.top_100_NNs.apply(lambda x: eval(x))
    df_top_titles=df_top_titles.drop_duplicates(subset=['title'])
    print('Number of hoax titles:', str(len(df_top_titles)))

    df_top_titles = df_top_titles.explode('top_100_NNs')
    df_top_titles = df_top_titles.reset_index(drop=True)
    df_top_titles = df_top_titles.rename(columns={'title':'hoax_title','top_100_NNs':'title'})
    df_top_titles=df_top_titles.drop_duplicates(subset=['title'])
    print('Number of Real titles:', str(len(df_top_titles)))

    df =pd.DataFrame()
    ts_list =[]
    #df['hoax_title'] = df_top_titles['hoax_title']
    df['title'] = df_top_titles.title.tolist()
    print(len(df))
    for i in df.title.tolist():
        ts_list.append(get_revision_timestamps(i))
        print('timestamp collected for: ',str(df.title.tolist().index(i)),'/',str(len(df.title.tolist())))
    df['revision_history'] = ts_list
    print(len(df))
    df = df.dropna(subset=['revision_history'])
    df = df.reset_index(drop=True)
    print('Number of Real titles with revision history:', str(len(df)))
    print('---------------------')
    df['recent_timestamp'] = df['revision_history'].apply(lambda x: list(x.keys())[-1])
    df = df.dropna(subset=['title'])
    for i in df.index:
        df.at[i,'text'] = clean_wiki_article('https://en.wikipedia.org/w/index.php?title='+df.at[i,'title']) #+'&oldid='+str(df['recent_timestamp'][i]
        df.at[i,'text'] = remove_old_revision(df.at[i,'text'])
        df.at[i,'text'] = remove_current_revision(df.at[i,'text'])
        df.at[i,'text'] = add_space_after_full_stop(df.at[i,'text'])
        print('text collected for: ',str(i),'/',str(len(df)))

    df['hoax_title'] = df_top_titles['hoax_title']
    df['data_source'] = df_top_titles['data_source']
    df['split'] = 'train'
    df['label'] = 0
    df= df.drop_duplicates(subset=['title'], keep='first')

    df.to_pickle(f'{output_path}/real_dedup.pkl',protocol=4)
    df.to_csv(f'{output_path}/real_dedup.csv',index=False)
    print('files saved, process compelte')
    print('---------------------')