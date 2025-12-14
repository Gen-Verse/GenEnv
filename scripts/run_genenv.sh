#!/bin/bash
# GenEnv Training Script
# 
# This script provides an example of how to run GenEnv training.
# Users should customize the paths and parameters below.

set -x

# ============================================================================
# Environment Setup
# ============================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_DEBUG=INFO
export RAY_TMPDIR=/tmp/ray_genenv
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Add GenEnv to Python path
export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

mkdir -p $RAY_TMPDIR

# ============================================================================
# Model Paths (USER CUSTOMIZATION REQUIRED)
# ============================================================================
# >>> Replace these with your actual model paths <<<
MODEL_PATH="/path/to/your/base/model"           # e.g., Qwen2.5-7B-Instruct
ENV_MODEL_PATH="/path/to/your/env/model"        # Environment LLM for task generation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --env-model)
            ENV_MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# ============================================================================
# Data Paths (USER CUSTOMIZATION REQUIRED)
# ============================================================================
# >>> Replace these with your actual dataset paths <<<
TRAIN_DATA="/path/to/your/train.parquet"
VAL_DATA="/path/to/your/validation.parquet"
OUTPUT_DIR="/path/to/save/checkpoints"

# ============================================================================
# Run GenEnv Training
# ============================================================================
python3 -u -m genenv.train \
    genenv.enable=True \
    genenv.filtering_k=0.1 \
    genenv.num_generations_per_prompt=4 \
    env_model_path=$ENV_MODEL_PATH \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=64 \
    data.val_batch_size=500 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_temperature=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='genenv' \
    trainer.experiment_name='genenv_training' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=20 \
    "${@:1}"

