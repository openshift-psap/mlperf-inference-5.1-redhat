# ============================================================================
# This script was generated and refactored with the help of AI (OpenAI GPT-4),
# with additional modifications and review by the author: <Naveen Miriyalu nmiriyal@redhat.com>.
#
# Disclaimer: This script is provided as-is, without warranty of any kind.
# Please review and test before using in production or submitting to MLPerf.
# ============================================================================

# Example: Offline scenario with local vLLM
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --max-num-seqs 1024 \
  --gpu-mem-util 0.9 \
  --batch-size 13368 \
  --test-mode performance \
  --scenario Offline \
  --num-gpus 1 \
  --pipeline-parallel-size 1 \
  --swap-space 4 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b \
  --max-num-batched-tokens 4096

# Example: Offline scenario with vLLM API
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --batch-size 13368 \
  --test-mode performance \
  --scenario Offline \
  --api-server-url http://localhost:8000 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b

# Example: MLPerf Server scenario with VLLMSingleSUTServer (multi-worker batching)
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --max-num-seqs 512 \
  --gpu-mem-util 0.9 \
  --batch-size 13368 \
  --test-mode performance \
  --scenario Server \
  --num-gpus 1 \
  --pipeline-parallel-size 1 \
  --swap-space 4 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b \
  --max-num-batched-tokens 4096 \
  --num-workers 4
