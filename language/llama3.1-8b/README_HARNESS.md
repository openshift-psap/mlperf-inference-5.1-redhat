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

### 4. Offline scenario with CUDA Graph optimization
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
  --cuda-graph-sizes 1 8 16 32 64 128 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b \
  --max-num-batched-tokens 4096
```

### 5. Server scenario with custom LLM config and advanced engine args
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
  --cuda-graph-sizes 1 8 16 32 \
  --long-prefill-token-threshold 2048 \
  --num-lookahead-slots 4 \
  --scheduler-delay-factor 0.1 \
  --preemption-mode recompute \
  --scheduling-policy fcfs \
  --enable-chunked-prefill \
  --block-size 16 \
  --kv-cache-dtype fp8 \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b \
  --max-num-batched-tokens 4096 \
  --num-workers 4
```

### 6. Offline scenario with generation config override
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
  --override-generation-config '{"temperature": 0.1, "top_p": 0.9, "max_tokens": 128}' \
  --log-level INFO \
  --output-log-dir ./ \
  --user-conf user.conf \
  --lg-model-name llama3_1-8b \
  --max-num-batched-tokens 4096
```

## Available vLLM Engine Options

### CUDA Graph and Performance Optimization
- `--cuda-graph-sizes` : List of sequence lengths for CUDA graph capture (e.g., "1 8 16 32 64 128")
- `--long-prefill-token-threshold` : Threshold for long prefill token processing
- `--num-lookahead-slots` : Number of lookahead slots for scheduling
- `--scheduler-delay-factor` : Delay factor for scheduler (0.0-1.0)
- `--preemption-mode` : Preemption mode (`recompute`, `swap`, or `None`)
- `--scheduling-policy` : Scheduling policy (`fcfs` or `priority`)
- `--enable-chunked-prefill` : Enable chunked prefill processing
- `--disable-chunked-mm-input` : Disable chunked multimodal input processing

### Memory and Cache Management
- `--block-size` : Block size for memory management (1, 8, 16, 32, 64, 128)
- `--kv-cache-dtype` : KV cache data type (`auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`)
- `--num-gpu-blocks-override` : Override number of GPU blocks
- `--enable-prefix-caching` : Enable prefix caching
- `--prefix-caching-hash-algo` : Prefix caching hash algorithm (`builtin` or `sha256`)
- `--cpu-offload-gb` : CPU offload size in GB
- `--calculate-kv-scales` : Calculate KV scales for quantization

### Model Configuration
- `--dtype` : Model data type (`auto`, `bfloat16`, `float16`, `float32`, `half`)
- `--quantization` : Quantization method (various options available)
- `--max-seq-len-to-capture` : Maximum sequence length to capture
- `--max-logprobs` : Maximum number of logprobs to return
- `--rope-scaling` : RoPE scaling configuration
- `--rope-theta` : RoPE theta parameter

### Generation Configuration
- `--generation-config` : Path to generation config file
- `--override-generation-config` : Override generation config as JSON string
- `--logits-processor-pattern` : Logits processor pattern
- `--guided-decoding-backend` : Guided decoding backend
- `--speculative-config` : Speculative decoding configuration

### Parallelism and Distribution
- `--tensor-parallel-size` : Tensor parallel size (number of GPUs)
- `--data-parallel-size` : Data parallel size
- `--enable-expert-parallel` : Enable expert parallelism
- `--max-parallel-loading-workers` : Maximum parallel loading workers
- `--distributed-executor-backend` : Distributed executor backend

### LoRA and Adapter Support
- `--enable-lora` : Enable LoRA support
- `--max-loras` : Maximum number of LoRAs
- `--max-lora-rank` : Maximum LoRA rank
- `--lora-dtype` : LoRA data type
- `--enable-prompt-adapter` : Enable prompt adapter
- `--max-prompt-adapters` : Maximum number of prompt adapters

### Multimodal and Special Features
- `--enable-reasoning` : Enable reasoning capabilities
- `--reasoning-parser` : Reasoning parser type
- `--limit-mm-per-prompt` : Limit multimodal content per prompt
- `--mm-processor-kwargs` : Multimodal processor kwargs
- `--enable-sleep-mode` : Enable sleep mode for power saving

### Advanced Configuration
- `--additional-config` : Additional configuration file
- `--compilation-config` : Compilation configuration
- `--kv-transfer-config` : KV transfer configuration
- `--kv-events-config` : KV events configuration
- `--use-v2-block-manager` : Use v2 block manager
- `--disable-log-stats` : Disable log statistics

### Device and Platform
- `--device` : Target device (`auto`, `cpu`, `cuda`, `hpu`, `neuron`, `tpu`, `xpu`)
- `--model-impl` : Model implementation (`auto`, `vllm`, `transformers`)
- `--load-format` : Model loading format

### Monitoring and Tracing
- `--show-hidden-metrics-for-version` : Show hidden metrics for specific version
- `--otlp-traces-endpoint` : OTLP traces endpoint
- `--collect-detailed-traces` : Collect detailed traces (`all`, `model`, `worker`, `None`)

## Notes
- For API mode, ensure the vLLM API server is running and accessible.
- For server scenario, adjust `--num-workers` and `--batch-size` for your hardware.
- All options can be listed with `python SUT_VLLM_SingleReplica.py --help`.

---
vLLM options
usage: SUT_VLLM_SingleReplica.py [-h] [--model MODEL] [--task {auto,classify,draft,embed,embedding,generate,reward,score,transcription}] [--tokenizer TOKENIZER]
                                 [--tokenizer-mode {auto,custom,mistral,slow}] [--trust-remote-code | --no-trust-remote-code]
                                 [--dtype {auto,bfloat16,float,float16,float32,half}] [--seed SEED] [--hf-config-path HF_CONFIG_PATH]
                                 [--allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH] [--revision REVISION] [--code-revision CODE_REVISION]
                                 [--rope-scaling ROPE_SCALING] [--rope-theta ROPE_THETA] [--tokenizer-revision TOKENIZER_REVISION] [--max-model-len MAX_MODEL_LEN]
                                 [--quantization {aqlm,auto-round,awq,awq_marlin,bitblas,bitsandbytes,compressed-tensors,deepspeedfp,experts_int8,fbgemm_fp8,fp8,gguf,gptq,gptq_bitblas,gptq_marlin,gptq_marlin_24,hqq,ipex,marlin,modelopt,modelopt_fp4,moe_wna16,neuron_quant,ptpc_fp8,qqq,quark,torchao,tpu_int8,None}]
                                 [--enforce-eager | --no-enforce-eager] [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE] [--max-logprobs MAX_LOGPROBS]
                                 [--disable-sliding-window | --no-disable-sliding-window] [--disable-cascade-attn | --no-disable-cascade-attn]
                                 [--skip-tokenizer-init | --no-skip-tokenizer-init] [--enable-prompt-embeds | --no-enable-prompt-embeds]
                                 [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]] [--disable-async-output-proc] [--config-format {auto,hf,mistral}]
                                 [--hf-token [HF_TOKEN]] [--hf-overrides HF_OVERRIDES] [--override-neuron-config OVERRIDE_NEURON_CONFIG]
                                 [--override-pooler-config OVERRIDE_POOLER_CONFIG] [--logits-processor-pattern LOGITS_PROCESSOR_PATTERN]
                                 [--generation-config GENERATION_CONFIG] [--override-generation-config OVERRIDE_GENERATION_CONFIG]
                                 [--enable-sleep-mode | --no-enable-sleep-mode] [--model-impl {auto,vllm,transformers}]
                                 [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer,runai_streamer_sharded,fastsafetensors}]
                                 [--download-dir DOWNLOAD_DIR] [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                                 [--ignore-patterns IGNORE_PATTERNS [IGNORE_PATTERNS ...]] [--use-tqdm-on-load | --no-use-tqdm-on-load]
                                 [--qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH] [--pt-load-map-location PT_LOAD_MAP_LOCATION]
                                 [--guided-decoding-backend {auto,guidance,lm-format-enforcer,outlines,xgrammar}]
                                 [--guided-decoding-disable-fallback | --no-guided-decoding-disable-fallback]
                                 [--guided-decoding-disable-any-whitespace | --no-guided-decoding-disable-any-whitespace]
                                 [--guided-decoding-disable-additional-properties | --no-guided-decoding-disable-additional-properties]
                                 [--enable-reasoning | --no-enable-reasoning] [--reasoning-parser {deepseek_r1,granite,qwen3}]
                                 [--distributed-executor-backend {external_launcher,mp,ray,uni,None}] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                                 [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--data-parallel-size DATA_PARALLEL_SIZE]
                                 [--data-parallel-size-local DATA_PARALLEL_SIZE_LOCAL] [--data-parallel-address DATA_PARALLEL_ADDRESS]
                                 [--data-parallel-rpc-port DATA_PARALLEL_RPC_PORT] [--data-parallel-backend DATA_PARALLEL_BACKEND]
                                 [--enable-expert-parallel | --no-enable-expert-parallel] [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                                 [--ray-workers-use-nsight | --no-ray-workers-use-nsight] [--disable-custom-all-reduce | --no-disable-custom-all-reduce]
                                 [--worker-cls WORKER_CLS] [--worker-extension-cls WORKER_EXTENSION_CLS]
                                 [--enable-multimodal-encoder-data-parallel | --no-enable-multimodal-encoder-data-parallel] [--block-size {1,8,16,32,64,128}]
                                 [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--swap-space SWAP_SPACE] [--kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2}]
                                 [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE] [--enable-prefix-caching | --no-enable-prefix-caching]
                                 [--prefix-caching-hash-algo {builtin,sha256}] [--cpu-offload-gb CPU_OFFLOAD_GB] [--calculate-kv-scales | --no-calculate-kv-scales]
                                 [--tokenizer-pool-size TOKENIZER_POOL_SIZE] [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
                                 [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG] [--limit-mm-per-prompt LIMIT_MM_PER_PROMPT]
                                 [--mm-processor-kwargs MM_PROCESSOR_KWARGS] [--disable-mm-preprocessor-cache | --no-disable-mm-preprocessor-cache]
                                 [--enable-lora | --no-enable-lora] [--enable-lora-bias | --no-enable-lora-bias] [--max-loras MAX_LORAS]
                                 [--max-lora-rank MAX_LORA_RANK] [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE] [--lora-dtype {auto,bfloat16,float16}]
                                 [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS [LONG_LORA_SCALING_FACTORS ...]] [--max-cpu-loras MAX_CPU_LORAS]
                                 [--fully-sharded-loras | --no-fully-sharded-loras] [--enable-prompt-adapter | --no-enable-prompt-adapter]
                                 [--max-prompt-adapters MAX_PROMPT_ADAPTERS] [--max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN]
                                 [--device {auto,cpu,cuda,hpu,neuron,tpu,xpu}] [--speculative-config SPECULATIVE_CONFIG]
                                 [--show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION] [--otlp-traces-endpoint OTLP_TRACES_ENDPOINT]
                                 [--collect-detailed-traces {all,model,worker,None} [{all,model,worker,None} ...]] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                                 [--max-num-seqs MAX_NUM_SEQS] [--max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS]
                                 [--max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS] [--cuda-graph-sizes CUDA_GRAPH_SIZES [CUDA_GRAPH_SIZES ...]]
                                 [--long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD] [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS]
                                 [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR] [--preemption-mode {recompute,swap,None}]
                                 [--num-scheduler-steps NUM_SCHEDULER_STEPS] [--multi-step-stream-outputs | --no-multi-step-stream-outputs]
                                 [--scheduling-policy {fcfs,priority}] [--enable-chunked-prefill | --no-enable-chunked-prefill]
                                 [--disable-chunked-mm-input | --no-disable-chunked-mm-input] [--scheduler-cls SCHEDULER_CLS]
                                 [--disable-hybrid-kv-cache-manager | --no-disable-hybrid-kv-cache-manager] [--kv-transfer-config KV_TRANSFER_CONFIG]
                                 [--kv-events-config KV_EVENTS_CONFIG] [--compilation-config COMPILATION_CONFIG] [--additional-config ADDITIONAL_CONFIG]
                                 [--use-v2-block-manager] [--disable-log-stats] [--dataset-path DATASET_PATH] [--num-samples NUM_SAMPLES] [--batch-size BATCH_SIZE]
                                 [--num-workers NUM_WORKERS] [--scenario {Offline,Server}] [--test-mode {performance,accuracy}]
                                 [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--output-log-dir OUTPUT_LOG_DIR] [--user-conf USER_CONF] [--audit-conf AUDIT_CONF]
                                 [--lg-model-name {llama3_1-8b,llama3_1-8b-interactive,test-model}] [--enable-profiler] [--profiler-dir PROFILER_DIR] [--enable-nvtx]
                                 [--print-timing] [--print-histogram] [--sort-by-length] [--sort-by-token-contents] [--print-sorted-tokens]
                                 [--api-server-url API_SERVER_URL] [--enable-metrics-csv] [--metrics-csv-path METRICS_CSV_PATH]

Author: <YOUR NAME HERE>
AI Assistance: OpenAI GPT-4 
