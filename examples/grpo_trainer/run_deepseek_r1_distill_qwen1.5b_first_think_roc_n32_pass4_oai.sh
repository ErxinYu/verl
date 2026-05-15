#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_deepseek_r1_distill_qwen1.5b_first_think_roc_n32_pass4_oai.sh
#
# 与 _roc_n32_pass4.sh 完全相同，唯一差异：
#   ++algorithm.grpo_roc.overlong_as_incorrect=True
#
# 超长响应（response_len >= max_response_length=16384）不再从 RoC 中丢弃，
# 而是将其 token_level_scores 清零，作为 incorrect 参与 GRPO 优势计算。
# 模型会收到明确的负梯度信号，预期能抑制响应长度持续增长。
#
# 对照组：run_deepseek_r1_distill_qwen1.5b_first_think_roc_n32_pass4.sh
#   （超长直接 drop，模型不收到针对超长的负梯度）
# ---------------------------------------------------------------------------

set -euo pipefail

export CUDA_HOME=/root/verl-env/cuda_home
PYTHON=/root/verl-env/bin/python3
PYTHON_VLLM=/root/gptoss-env/bin/python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_PATH="${SCRIPT_DIR}/first_think_pass4_reward.py"

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPERIMENT_NAME="deepseek_r1_grpo_first_think_roc_n32_pass4_oai"
DEFAULT_LOCAL_DIR=/mnt/ckpt/${EXPERIMENT_NAME}

mkdir -p "${DEFAULT_LOCAL_DIR}"
LOG_FILE="${DEFAULT_LOCAL_DIR}/log.txt"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script started. Log → ${LOG_FILE}"

# ── vllm ref-model server (GPU 7) ───────────────────────────────────────────
VLLM_GPU=7
VLLM_PORT=8001
VLLM_URL="http://localhost:${VLLM_PORT}"
VLLM_LOG="${DEFAULT_LOCAL_DIR}/vllm_server.log"

echo "[setup] Starting vllm ref-model server on GPU ${VLLM_GPU} (port ${VLLM_PORT})…"
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${PYTHON_VLLM}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --served-model-name "default" \
    >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[cleanup] Stopping vllm server (pid ${VLLM_PID})…"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[setup] Waiting for vllm server to become ready…"
for i in $(seq 1 60); do
    if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
        echo "[setup] vllm server ready after $((i * 2)) s."
        break
    fi
    sleep 2
    if [ "${i}" -eq 60 ]; then
        echo "[error] vllm server did not start within 120 s. Aborting."
        exit 1
    fi
done

# ── GRPO training (GPUs 0–6) ────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
N_TRAIN_GPUS=7

"${PYTHON}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/code/MWPBench/result/qwen1_5_dapo_new/dapo_rl_train.jsonl \
    data.val_files=/mnt/data/dapo/dapo_test_verl.jsonl \
    data.train_batch_size=112 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=56 \
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
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    ++algorithm.grpo_roc.enable=True \
    ++algorithm.grpo_roc.target_n=16 \
    ++algorithm.grpo_roc.quality_key=pass_rate \
    ++algorithm.grpo_roc.overlong_as_incorrect=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_dapo' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_TRAIN_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.total_epochs=3 \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    ++reward.custom_reward_function.reward_kwargs.vllm_url="${VLLM_URL}" \
    ++reward.custom_reward_function.reward_kwargs.pass_at_k=4 \
    ++reward.custom_reward_function.reward_kwargs.temperature=0.6 \
    ++reward.custom_reward_function.reward_kwargs.max_tokens=4096 \
    '++trainer.custom_metric_keys=["pass_rate","resp_tokens","first_think_tokens","original_acc","vllm_error"]' \
    "$@" \
    ${TOTAL_STEPS:+"++trainer.total_training_steps=${TOTAL_STEPS}"}
