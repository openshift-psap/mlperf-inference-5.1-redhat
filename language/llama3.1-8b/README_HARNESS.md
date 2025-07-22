# ============================================================================
# README_HARNESS.md
#
# This documentation was generated and refactored with the help of AI (OpenAI GPT-4),
# with additional modifications and review by the author: <Naveen Miriyalu nmiriyal@redhat.com>
#
# Disclaimer: This documentation is provided as-is, without warranty of any kind.
# Please review and test before using in production or submitting to MLPerf.
# ============================================================================

# MLPerf vLLM Harness Usage Guide

This harness supports running vLLM models with MLPerf Loadgen in both offline and server scenarios. It supports:
- Local vLLM (direct model loading)
- vLLM API (remote inference)
- MLPerf Server scenario with async batching and multi-worker support

## Command-Line Options

Options are grouped logically for clarity:

### Model and Data
- `--model-name` : Name or path of the model to load (e.g., HuggingFace repo or local path)
- `--dataset-path` : Path to the processed dataset pickle file
- `--num-samples` : Number of samples/prompts to use
- `--max-model-len` : Maximum sequence length for the model
- `--max-num-seqs` : Maximum number of sequences processed simultaneously
- `--max-num-batched-tokens` : Maximum number of batched tokens for vLLM batching

### Performance and Parallelism
- `--batch-size` : Batch size for worker(s)
- `--num-workers` : Number of worker threads (server scenario)
- `--num-gpus` : Number of GPUs (tensor parallel size)
- `--pipeline-parallel-size` : Pipeline parallel size
- `--swap-space` : Swap space parameter for vLLM
- `--gpu-mem-util` : GPU memory utilization factor

### Scenario and Mode
- `--scenario` : MLPerf scenario (`Offline` or `Server`)
- `--test-mode` : Test mode (`performance` or `accuracy`)
- `--api-server-url` : URL of vLLM API server (for API mode)

### Logging and Output
- `--log-level` : Logging level (DEBUG, INFO, etc.)
- `--output-log-dir` : Directory for logs
- `--user-conf` : User config for LoadGen
- `--lg-model-name` : Model name for LoadGen

### Advanced/Debug
- `--enable-profiler` : Enable torch profiler
- `--profiler-dir` : Directory for profiler traces
- `--enable-nvtx` : Enable NVTX profiling
- `--print-histogram` : Print input length histograms
- `--sort-by-length` : Sort queries by input length
- `--sort-by-token-contents` : Sort queries by token contents
- `--print-sorted-tokens` : Print sorted tokens
- `--print-timing` : Print timing statistics
- `--enable-metrics-csv` : Enable periodic metrics collection (API only)
- `--metrics-csv-path` : Path for metrics CSV (API only)

## Example Usage

### 1. Offline scenario with local vLLM
```sh
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --max-model-len 2048 \
  --max-num-seqs 512 \
  --gpu-mem-util 0.9 \
  --batch-size 32 \
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
```

### 2. Offline scenario with vLLM API
```sh
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --batch-size 32 \
  --test-mode performance \
  --scenario Offline \
  --api-server-url http://localhost:8000 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b
```

### 3. MLPerf Server scenario with VLLMSingleSUTServer (multi-worker batching)
```sh
python SUT_VLLM_SingleReplica.py \
  --model-name <MODEL_NAME> \
  --dataset-path <DATASET_PATH> \
  --num-samples 13368 \
  --max-model-len 2048 \
  --max-num-seqs 512 \
  --gpu-mem-util 0.9 \
  --batch-size 32 \
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
```

## Notes
- For API mode, ensure the vLLM API server is running and accessible.
- For server scenario, adjust `--num-workers` and `--batch-size` for your hardware.
- All options can be listed with `python SUT_VLLM_SingleReplica.py --help`.

---

Author: <YOUR NAME HERE>
AI Assistance: OpenAI GPT-4 