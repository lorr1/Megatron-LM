import os
import sys
import time
import yaml
import wandb
import argparse
import pandas as pd
import datetime
from datasets import load_dataset

sys.path.append('../')
from  tools import preprocess_data

def download_dataset(dataset, datafile=None):
    if not datafile:
        return load_dataset(dataset)
    return load_dataset(dataset, datafile)

def save_loose_json(hf_dataset, corpus_path, split='train'):
    if split == 'all':
        dataset_sizes = {}
        datasets = []
        for splt in ['train', 'test', 'validation']:
            hf_ds_obj = hf_dataset[splt]
            df = pd.DataFrame(hf_ds_obj)
            # filter out all empty strings
            fltr = df['text'] != ""
            df = df[fltr]
            datasets.append(df)
            dataset_sizes[splt] = len(df)
        df_joint = pd.concat(datasets, ignore_index=True)
        df_joint.to_json(corpus_path, orient='records', lines=True)
        wandb.log(dataset_sizes)
    else:        
        hf_ds_obj = hf_dataset[split]
        df = pd.DataFrame(hf_ds_obj)
        # filter out all empty strings
        fltr = df['text'] != ""
        df = df[fltr]
        df.to_json(corpus_path, orient='records', lines=True)

def preprocess(megatron_params):
    params = ['--append-eod']
    for key, value in megatron_params.items():
        params.append(key)
        params.append(str(value))
    start = time.time()
    preprocess_data.main(params)
    end = time.time()
    return (end-start)

def main():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--config_file', default=None, help='path to dataset config file')
    args = parser.parse_args()
    with open(args.config_file) as cf:
        template = yaml.load(cf, Loader=yaml.SafeLoader)
    wandb.init(project=template['wandb-project'], config=template)
    print("Loading dataset...")
    ds = load_dataset(template['dataset']['dataset-name'], template['dataset']['datafile'])
    print("Creating loose json file...")
    corpus_filepath = save_loose_json(ds, template['megatron_preprocess']['--input'], template['dataset']['split'])
    print("Preprocessing data...")
    time_to_preprocess = preprocess(template['megatron_preprocess'])
    print("Preprocess complete!")
    wandb.log({'time to complete megatron preprocess' : time_to_preprocess})


if __name__=="__main__":
    main()

