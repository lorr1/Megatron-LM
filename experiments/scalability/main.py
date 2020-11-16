import datetime
import os
from argparse import ArgumentParser

import pandas as pd
import yaml
import subprocess
from quinine import *
from quinine.common.utils import difference

# Create an argument parser to input the sweep
parser = ArgumentParser(description='Sweep for scalability analysis.')
parser.add_argument('--sweep', help='Path to sweep config file.')
args = parser.parse_args()

quinsweep = QuinSweep(sweep_config_path=args.sweep)

# Make a directory for the sweep configs
now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
config_dir = os.path.join(os.path.dirname(args.sweep),
                          os.path.basename(args.sweep).replace(".yaml", "") + f"_configs_{now}")
os.makedirs(config_dir)

# Extract the configs in the sweep
quinfigs = list(quinsweep)

# Save each sweep config
for i, quinfig in enumerate(quinfigs):
    yaml.dump(quinfig, open(os.path.join(config_dir, f'{i}.yaml'), 'w'))

# Save a summary of the sweep
with open(os.path.join(config_dir, 'summary.txt'), 'w') as outfile:
    pd.DataFrame(difference(*quinfigs)).to_string(outfile)


# Run
for i in range(1):
    subprocess.run(['MKL_THREADING_LAYER=TBB', 'python', '-m', 'pretrain_gpt2', '--config', f'{config_dir}/{i}.yaml'])
