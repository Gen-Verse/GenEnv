#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 GenEnv Authors
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
#
# This file is adapted from verl (https://github.com/volcengine/verl)

"""
Main entry point for GenEnv training.

Usage:
    python -m genenv.train \\
        genenv.enable=True \\
        genenv.filtering_k=0.1 \\
        genenv.num_generations_per_prompt=4 \\
        env_model_path=/path/to/env/model \\
        ...

See configs/genenv_config.yaml for full configuration options.
"""

import os
import ray
import hydra
from omegaconf import OmegaConf

from genenv.trainer.genenv_trainer import run_genenv_training
from genenv.utils.reward_functions import RewardManager


@hydra.main(config_path='../configs', config_name='genenv_config', version_base=None)
def main(config):
    """Main entry point for GenEnv training."""
    
    # Set environment variables
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth1')
    
    # Initialize Ray
    if not ray.is_initialized():
        runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'NCCL_SOCKET_IFNAME': 'eth1',
                'GLOO_SOCKET_IFNAME': 'eth1'
            }
        }
        ray.init(
            address=config.trainer.ray_address,
            runtime_env=runtime_env,
            _temp_dir=os.environ.get("RAY_TMPDIR")
        )

    # Print configuration
    from pprint import pprint
    print("=" * 60)
    print("GenEnv Configuration:")
    print("=" * 60)
    pprint(OmegaConf.to_container(config, resolve=True))
    print("=" * 60)

    # Run training
    if config.genenv.enable:
        print("[GenEnv] Co-training mode enabled")
        ray.get(run_genenv_task.remote(config))
    else:
        print("[GenEnv] Standard GRPO training mode")
        ray.get(standard_training_task.remote(config))


@ray.remote
def run_genenv_task(config):
    """Ray remote task for GenEnv co-training."""
    run_genenv_training(config)


@ray.remote
def standard_training_task(config):
    """Ray remote task for standard GRPO training (without co-training)."""
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    
    from pprint import pprint
    from omegaconf import OmegaConf
    
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    resource_pool_spec = {
        'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: 'global_pool',
        Role.RefPolicy: 'global_pool',
    }

    # >>> USER CUSTOMIZATION: Replace with your reward function <<<
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

