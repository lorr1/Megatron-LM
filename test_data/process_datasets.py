"""
process_data.py
usage:
    modify the config file preprocess_config.yaml to suit the needs of your dataset
    then run this follows as follows: python process_data.py --config_file preprocess_config.yaml
"""
import os
import sys
import time
import yaml
import wandb
import pandas as pd
import argparse

sys.path.append('../')
from tools import preprocess_data

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
    parser.add_argument('--config_file', default='preprocess_config.yaml', help='path to dataset config file')
    args = parser.parse_args()
    with open(args.config_file) as cf:
        template = yaml.load(cf, Loader=yaml.SafeLoader)
    wandb.init(project=template['wandb-project'], config=template)
    print("starting pre-processing...")
    time_to_preprocess = preprocess(template['megatron_preprocess'])
    print("Preprocess complete!")
    wandb.log({'time to complete megatron preprocess' : time_to_preprocess})

if __name__=="__main__":
    main()
