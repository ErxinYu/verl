#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_deepseek_r1_distill_qwen1.5b_first_think.sh
#
# Training script with First-Think Efficiency Reward:
#   score = correctness + first_think_weight * (first_think_len / total_think_len)
#
# "First thinking" = text between <think> and the first backtracking word
# (But, Wait, Alternatively, Actually, …).  Longer first-think → higher bonus
# within a group; GRPO's advantage normalisation handles within-group ranking.
#
# Key new flags vs. the baseline script:
#   reward.custom_reward_function.path      – path to our reward module
#   reward.custom_reward_function.name      – function name inside that module
#   reward.custom_reward_function.reward_kwargs.first_think_weight
#                                           – weight of first-think bonus (default 0.5)
#   trainer.custom_metric_keys              – extra non_tensor_batch fields to log each step
#                                             logged as train/<key>/{mean,max,min} in wandb
#   trainer.experiment_name                 – updated to reflect new reward
# ---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_HOME=/root/verl-env/cuda_home

# Absolute path to the custom reward file so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_PATH="${SCRIPT_DIR}/first_think_reward.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/code/MWPBench/result/qwen1_5_dapo_new/dapo_rl_train.jsonl \
    data.val_files=/mnt/data/dapo/dapo_test_verl.jsonl \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=18 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_dapo' \
    trainer.experiment_name='deepseek_r1_grpo_first_think_w0.5' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir=/mnt/ckpt/deepseek_r1_grpo_first_think_0422 \
    trainer.total_epochs=3 \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    ++reward.custom_reward_function.reward_kwargs.first_think_weight=0.5 \
    '++trainer.custom_metric_keys=["first_think_len","total_think_len","first_think_ratio","first_think_bonus","acc","correctness"]' \
    $@
