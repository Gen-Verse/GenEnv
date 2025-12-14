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
# with modifications for GenEnv co-training framework.

"""
GenEnv Trainer: Co-training framework for Agent and Environment LLMs.

This trainer implements the GenEnv algorithm which alternates between:
1. Agent Training: Train the agent policy using GRPO on the current dataset
2. Environment Generation: Generate new challenging tasks using the Env LLM
3. Dataset Augmentation: Merge new tasks with the original dataset

The key innovation is the adaptive curriculum where the Environment LLM
learns to generate tasks at the boundary of the Agent's capability.
"""

import os
import gc
import time
import tempfile
import numpy as np
import pandas as pd
import ray
import torch
from typing import Dict, List, Any, Optional
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from genenv.utils.reward_functions import RewardManager


@ray.remote(num_gpus=8)
class EnvGeneratorWorker:
    """
    Environment Generator Worker using vLLM for efficient inference.
    
    This worker is responsible for generating new training tasks based on
    existing tasks and their solutions. Users should customize the generation
    prompt template in `_build_generation_prompt` method.
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 4):
        """
        Initialize the Environment Generator.
        
        Args:
            model_path: Path to the Environment LLM (e.g., Qwen2.5-7B-Instruct)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.75,
            max_model_len=8192
        )
        
    def generate(self, prompts: List[str], n: int = 1, max_tokens: int = 8192) -> List[str]:
        """
        Generate new tasks from the given prompts.
        
        Args:
            prompts: List of formatted prompts for task generation
            n: Number of generations per prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated task descriptions
        """
        sampling_params = SamplingParams(n=n, temperature=1.0, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


def run_genenv_training(config):
    """
    Main entry point for GenEnv co-training.
    
    This function orchestrates the alternating training between Agent and Environment.
    
    Args:
        config: Hydra configuration object containing all training parameters
        
    Note:
        Users need to customize:
        - reward_fn: Your domain-specific reward function
        - env_prompt_template: How to prompt the Env LLM for task generation
        - task_parsing: How to extract new tasks from Env LLM outputs
    """
    from transformers import AutoTokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    
    print("[GenEnv] Starting co-training loop...")
    
    # Initialize tokenizers
    agent_model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    agent_tokenizer = AutoTokenizer.from_pretrained(agent_model_path, trust_remote_code=True)
    
    env_model_path = copy_local_path_from_hdfs(config.env_model_path)
    env_tokenizer = AutoTokenizer.from_pretrained(env_model_path, trust_remote_code=True)
    
    current_train_file = config.data.train_files
    
    for epoch in range(config.trainer.total_epochs):
        print(f"\n{'='*60}")
        print(f"[GenEnv] Epoch {epoch+1}/{config.trainer.total_epochs}")
        print(f"{'='*60}")
        
        # =====================================================================
        # Phase 1: Agent Training
        # =====================================================================
        print("\n[GenEnv] Phase 1: Agent Training")
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }
        resource_pool_spec = {'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: 'global_pool',
            Role.RefPolicy: 'global_pool',
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # >>> USER CUSTOMIZATION: Replace with your reward function <<<
        reward_fn = RewardManager(tokenizer=agent_tokenizer, num_examine=0)
        val_reward_fn = RewardManager(tokenizer=agent_tokenizer, num_examine=1)
        
        # Update config with current train file
        agent_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        agent_config.data.train_files = current_train_file
        
        # Load from checkpoint if not first epoch
        if epoch > 0:
            actor_dir = os.path.join(config.trainer.default_local_dir, 'actor')
            if os.path.exists(actor_dir):
                steps = []
                for d in os.listdir(actor_dir):
                    if d.startswith('global_step_'):
                        try:
                            steps.append(int(d.split('_')[-1]))
                        except:
                            pass
                if steps:
                    latest_step = max(steps)
                    latest_ckpt = os.path.join(actor_dir, f'global_step_{latest_step}')
                    print(f"[GenEnv] Loading checkpoint from: {latest_ckpt}")
                    agent_config.actor_rollout_ref.model.path = latest_ckpt
        
        # Create and run trainer
        agent_trainer = RayPPOTrainer(
            config=agent_config,
            tokenizer=agent_tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        agent_trainer.init_workers()
        agent_trainer.fit(epochs=1)
        
        # =====================================================================
        # Phase 2: Evaluate Agent Performance on Current Dataset
        # =====================================================================
        print("\n[GenEnv] Phase 2: Evaluating Agent Performance")
        
        prompt_accuracies = _evaluate_agent_performance(
            agent_trainer=agent_trainer,
            agent_tokenizer=agent_tokenizer,
            config=config,
            current_train_file=current_train_file,
            reward_fn=reward_fn
        )
        
        # Save checkpoint
        agent_trainer._save_checkpoint()
        
        # Cleanup agent resources
        _cleanup_trainer(agent_trainer)
        
        # =====================================================================
        # Phase 3: Environment Generation
        # =====================================================================
        print("\n[GenEnv] Phase 3: Environment Generation")
        
        # Filter prompts based on agent performance
        filtered_prompts = _filter_prompts(
            prompt_accuracies=prompt_accuracies,
            filtering_k=config.genenv.filtering_k
        )
        
        # Generate new tasks
        new_dataset = _generate_new_tasks(
            filtered_prompts=filtered_prompts,
            env_model_path=env_model_path,
            env_tokenizer=env_tokenizer,
            config=config,
            epoch=epoch
        )
        
        # =====================================================================
        # Phase 4: Dataset Augmentation
        # =====================================================================
        print("\n[GenEnv] Phase 4: Dataset Augmentation")
        
        if new_dataset:
            current_train_file = _augment_dataset(
                new_dataset=new_dataset,
                original_train_file=config.data.train_files,
                config=config,
                epoch=epoch
            )
        else:
            print("[GenEnv] Warning: No new data generated. Reusing original dataset.")
            current_train_file = config.data.train_files
    
    print("\n[GenEnv] Training completed!")


def _evaluate_agent_performance(
    agent_trainer: RayPPOTrainer,
    agent_tokenizer,
    config,
    current_train_file: str,
    reward_fn
) -> Dict[str, Dict]:
    """
    Evaluate agent performance on the current training dataset.
    
    Returns:
        Dictionary mapping prompt -> {'scores': [...], 'gt': ground_truth}
    """
    from torch.utils.data import DataLoader
    
    eval_dataset = RLHFDataset(
        parquet_files=current_train_file,
        tokenizer=agent_tokenizer,
        prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.data.train_batch_size,
        collate_fn=collate_fn
    )
    
    prompt_accuracies = {}
    n_gen = config.genenv.num_generations_per_prompt
    
    for batch_dict in eval_dataloader:
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']
        position_ids = batch_dict['position_ids']
        batch_size = len(input_ids)
        
        # Decode prompts
        prompts = []
        for k in range(batch_size):
            valid_ids = input_ids[k][attention_mask[k] == 1]
            prompts.append(agent_tokenizer.decode(valid_ids, skip_special_tokens=False))
        
        ground_truths = [item['ground_truth'] for item in batch_dict['reward_model']]
        
        # Expand batch for multiple generations
        expanded_input_ids = input_ids.unsqueeze(1).repeat(1, n_gen, 1).view(batch_size * n_gen, -1)
        expanded_attention_mask = attention_mask.unsqueeze(1).repeat(1, n_gen, 1).view(batch_size * n_gen, -1)
        expanded_position_ids = position_ids.unsqueeze(1).repeat(1, n_gen, 1).view(batch_size * n_gen, -1)
        
        expanded_rm = []
        for i in range(batch_size):
            expanded_rm.extend([batch_dict['reward_model'][i]] * n_gen)
        
        batch = DataProto.from_single_dict({
            'input_ids': expanded_input_ids,
            'attention_mask': expanded_attention_mask,
            'position_ids': expanded_position_ids,
            'reward_model': np.array(expanded_rm, dtype=object)
        })
        
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, agent_trainer.actor_rollout_wg.world_size)
        generated_batch_padded = agent_trainer.actor_rollout_wg.generate_sequences(batch_padded)
        generated_batch = unpad_dataproto(generated_batch_padded, pad_size)
        
        if not isinstance(generated_batch, DataProto):
            generated_batch = DataProto(
                batch=generated_batch.batch,
                non_tensor_batch=generated_batch.non_tensor_batch,
                meta_info=generated_batch.meta_info
            )
        
        rewards = reward_fn(generated_batch).sum(-1).tolist()
        
        for i in range(batch_size):
            prompt_text = prompts[i]
            if prompt_text not in prompt_accuracies:
                prompt_accuracies[prompt_text] = {'scores': [], 'gt': ground_truths[i]}
            prompt_rewards = rewards[i * n_gen: (i + 1) * n_gen]
            prompt_accuracies[prompt_text]['scores'].extend(prompt_rewards)
    
    return prompt_accuracies


def _filter_prompts(prompt_accuracies: Dict, filtering_k: float) -> Dict:
    """
    Filter out prompts that are too easy (always solved) or too hard (never solved).
    
    The GenEnv algorithm focuses on prompts at the boundary of the agent's capability.
    """
    num_prompts = len(prompt_accuracies)
    num_to_remove = int(num_prompts * filtering_k)
    
    prompt_avg_scores = {p: np.mean(v['scores']) for p, v in prompt_accuracies.items()}
    sorted_prompts = sorted(prompt_avg_scores.items(), key=lambda item: item[1])
    
    # Remove easiest and hardest prompts
    prompts_to_remove = set([p for p, s in sorted_prompts[:num_to_remove]])
    prompts_to_remove.update([p for p, s in sorted_prompts[-num_to_remove:]])
    
    filtered_prompts = {p: v for p, v in prompt_accuracies.items() if p not in prompts_to_remove}
    print(f"[GenEnv] Filtered {len(prompts_to_remove)} prompts. Remaining: {len(filtered_prompts)}")
    
    return filtered_prompts


def _generate_new_tasks(
    filtered_prompts: Dict,
    env_model_path: str,
    env_tokenizer,
    config,
    epoch: int
) -> List[Dict]:
    """
    Generate new training tasks using the Environment LLM.
    
    >>> USER CUSTOMIZATION REQUIRED <<<
    Modify the prompt template and parsing logic for your specific domain.
    """
    print(f"[GenEnv] Generating new tasks using Env LLM...")
    
    env_worker = EnvGeneratorWorker.remote(env_model_path)
    
    prompts_list = []
    original_infos = []
    
    for prompt, data in filtered_prompts.items():
        # >>> USER CUSTOMIZATION: Modify this prompt template <<<
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Based on the original task and its correct answer, create a new, similar task and provide its correct answer. Format your response as:\nNew Task: <task>\nNew Answer: <answer>"},
            {"role": "user", "content": f"Original Task: {prompt}\nOriginal Answer: {data['gt']}"}
        ]
        env_prompt = env_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(env_prompt)
        original_infos.append(data)
    
    print(f"[GenEnv] Sending {len(prompts_list)} prompts to Env LLM...")
    generated_texts = ray.get(env_worker.generate.remote(prompts_list, n=1, max_tokens=8192))
    
    new_dataset = []
    import re
    
    for i, text in enumerate(generated_texts):
        try:
            # >>> USER CUSTOMIZATION: Modify parsing logic <<<
            match = re.search(r"New Task:(.*?)New Answer:(.*)", text, re.DOTALL | re.IGNORECASE)
            
            if match:
                new_task = match.group(1).strip()
                new_answer = match.group(2).strip()
                
                new_dataset.append({
                    config.data.prompt_key: [{"role": "user", "content": new_task}],
                    "reward_model": {"ground_truth": new_answer}
                })
        except Exception as e:
            print(f"[GenEnv] Warning: Error parsing output: {e}")
            continue
    
    ray.kill(env_worker)
    print(f"[GenEnv] Generated {len(new_dataset)} new tasks")
    
    return new_dataset


def _augment_dataset(
    new_dataset: List[Dict],
    original_train_file: str,
    config,
    epoch: int
) -> str:
    """
    Merge new generated tasks with the original training dataset.
    """
    new_df = pd.DataFrame(new_dataset)
    original_df = pd.read_parquet(original_train_file)
    
    # Ensure all required columns exist
    for col in original_df.columns:
        if col not in new_df.columns:
            if col == 'extra_info':
                new_df[col] = [{'id': f'genenv_{epoch}_{i}', 'index': i, 'split': 'env_generated'}
                               for i in range(len(new_df))]
            elif col == 'data_source':
                new_df[col] = 'genenv_generated'
            else:
                new_df[col] = None
    
    combined_df = pd.concat([original_df, new_df], ignore_index=True)
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".parquet") as tmp:
        combined_df.to_parquet(tmp.name)
        new_train_file = tmp.name
    
    print(f"[GenEnv] Dataset augmented: {len(original_df)} + {len(new_df)} = {len(combined_df)} samples")
    
    return new_train_file


def _cleanup_trainer(trainer):
    """Release trainer resources."""
    from ray.util.placement_group import remove_placement_group
    
    print("[GenEnv] Releasing trainer resources...")
    
    for wg in trainer.wg_dicts:
        for worker in wg.workers:
            ray.kill(worker)
    
    if hasattr(trainer, 'resource_pool_manager'):
        for pool_name, pool in trainer.resource_pool_manager.resource_pool_dict.items():
            if hasattr(pool, 'pgs') and pool.pgs:
                for pg in pool.pgs:
                    try:
                        remove_placement_group(pg)
                    except Exception:
                        pass
    
    del trainer
    gc.collect()
    time.sleep(5)

