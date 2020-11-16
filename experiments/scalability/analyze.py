import os

import wandb
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import namedtuple
import pandas as pd
from quinine.common.utils import difference
from types import SimpleNamespace


def fetch_all_wandb_run_ids(entity, project, filters=None, wandb_api=None):
    """
    Fetch all runs inside a Weights and Biases project.
    """
    if wandb_api is None:
        wandb_api = wandb.Api()
    wandb_path = f'{entity}/{project}/*'
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
        failed=False if run.state == 'finished' else True
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
    for run_id in run_ids
]

infos = [
    load_info_from_run(run)
    for run in runs
]

df = pd.concat([
    # Parameters that are different across the runs
    pd.DataFrame(difference(*[run.config for run in runs])),
    # Run information
    pd.DataFrame([run.path for run in runs], columns=['entity', 'project', 'run_id']),
    # Number of GPUs
    pd.DataFrame([run.config['nproc_per_node'] for run in runs], columns=['nproc_per_node']),
    # Memory usage
    pd.DataFrame([info.memory for info in infos]),
    # Time taken per iteration
    pd.DataFrame([np.mean(info.time_per_iter[1:]) for info in infos], columns=['time_per_iter']),
    # Whether run was successful
    pd.DataFrame([info.failed for info in infos], columns=['failed']),
], axis=1)

with open('report.txt', 'w') as f:
    print(df.to_string(f))
# Fix l-1 parameters, vary lth