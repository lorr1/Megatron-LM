# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import wandb
import os
from .package_info import (
    __description__,
    __contact_names__,
    __url__,
    __download_url__,
    __keywords__,
    __license__,
    __package_name__,
    __version__,
)

from .global_vars import get_args
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize  import initialize_megatron

def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def wand_init_0(args):
    """If distributed is initialized init wandb."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(f"Saving wandb to {os.path.join(args.save)}")
            wandb.init(config=args, name=args.name, project=args.project, entity=args.entity, group=args.group, job_type=args.job_type,
                       dir=args.save, save_code=True)#, sync_tensorboard=True)
            wandb.tensorboard.patch(pytorch=True)
    else:
        print(f"Saving wandb to {os.path.join(args.save)}")
        wandb.init(config=args, name=args.name, project=args.project, entity=args.entity, group=args.group, job_type=args.job_type,
                   dir=args.save, save_code=True)#, sync_tensorboard=True)
        wandb.tensorboard.patch(pytorch=True)
