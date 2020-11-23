import os

import wandb
import cytoolz as tz
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import namedtuple
import pandas as pd
from quinine.common.utils import difference
from types import SimpleNamespace
from tqdm.auto import tqdm


def fetch_all_wandb_run_ids(entity, project, filters=None, wandb_api=None):
    """
    Fetch all runs inside a Weights and Biases project.
    """
    if wandb_api is None:
        wandb_api = wandb.Api()
    wandb_path = f'{entity}/{project}'
    runs = wandb_api.runs(wandb_path, filters={} if not filters else filters)
    return [run.id for run in runs]


def load_wandb_run(run_id,
                   entity,
                   project,
                   wandb_api=None):
    """
    Load up a run from Weights and Biases.
    """
    if wandb_api is None:
        wandb_api = wandb.Api()

    # Path to Weights and Biases run
    wandb_path = f'{entity}/{project}/{run_id}'
    # Load the run
    run = wandb_api.run(wandb_path)
    return run


def load_info_from_run(run):
    memory = {}
    time_per_iter = []

    for i, row in enumerate(run.scan_history()):
        step = row['_step']
        if step == 0:
            memory['param_count'] = row['params/total_count']
            memory['param_mb'] = row['params/total_mb_grad']
        else:
            try:
                if step == 1:
                    memory['memory_allocated'] = row['info/memory_allocated']
                    memory['memory_max_allocated'] = row['info/memory_max_allocated']
                    memory['memory_reserved'] = row['info/memory_reserved']
                    memory['memory_max_reserved'] = row['info/memory_max_reserved']
                time_per_iter.append(row['info/time_per_iter'])
            except KeyError:
                continue

    return SimpleNamespace(
        memory=memory,
        time_per_iter=time_per_iter,
        failed=False if run.state == 'finished' or run.id in {'151q6uwk', '3r1pr73v', '1pujl0iq', 'do0609f3',
                                                              '2dijkopv', '2mthlxld', 'q5hnq0h4'} else True
        # (karan) 151q6uwk crashed due to user error, finished successfully
    )


# Configure
entity = 'hazy-research'
project = 'gpt-rep-test'
filters = {'group': 'scalability-analysis'}

# Load runs
api = wandb.Api()
run_ids = fetch_all_wandb_run_ids(
    entity=entity,
    project=project,
    filters={'group': 'scalability-analysis'},
    wandb_api=api
)
print(f"> Found {len(run_ids)} runs.")
runs = [
    load_wandb_run(
        run_id,
        entity=entity,
        project=project,
        wandb_api=api
    )
    for run_id in tqdm(run_ids, "Fetching runs")
]

infos = [
    load_info_from_run(run)
    for run in tqdm(runs, "Loading info")
]

# Find the differences between all the run configs
config_diffs = difference(*[run.config for run in runs])
# Clean up the parameter names (tuple -> str)
config_diffs = list(map(lambda d: tz.keymap(lambda k: k[0] if len(k) == 1 else k, d), config_diffs))

# Create a single dataframe containing run information
df = pd.concat([
    # Run information
    pd.DataFrame([run.path for run in runs], columns=['entity', 'project', 'run_id']),
    # Parameters that are different across runs
    pd.DataFrame(config_diffs),
    # Data parallel size
    pd.DataFrame([int(run.config['nproc_per_node'] / run.config['model_parallel_size']) for run in runs],
                 columns=['data_parallel_size']),
    # Global batch size
    pd.DataFrame([int(run.config['nproc_per_node'] * run.config['batch_size'] / run.config['model_parallel_size'])
                  for run in runs],
                 columns=['total_batch_size']),
    # Memory usage
    pd.DataFrame([info.memory for info in infos]),
    # Time taken per iteration
    pd.DataFrame([np.mean(info.time_per_iter[1:]) for info in infos], columns=['time_per_iter']),
    # Whether run was successful
    pd.DataFrame([info.failed for info in infos], columns=['failed']),
], axis=1)

# Add more columns
df = pd.concat([
    df,
    # Global memory usage
    pd.DataFrame(df[["nproc_per_node", "memory_max_reserved"]].product(axis=1, skipna=False),
                 columns=['global_memory_max_reserved']),
], axis=1)

# Save the dataframe to file
with open('experiments/scalability/report.txt', 'w') as f:
    df.to_string(f)
df.to_pickle('experiments/scalability/analysis.p')

# Analysis columns
knobs = [
    'batch_size',
    'num_layers',
    'hidden_size',
    'nproc_per_node',
    'num_attention_heads',
    'total_batch_size',
]
observations = [
    'param_count',
    'param_mb',
    'memory_allocated',
    'memory_max_allocated',
    'memory_reserved',
    'memory_max_reserved',
    'time_per_iter',
    'failed'
]
