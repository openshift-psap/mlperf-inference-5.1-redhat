### ======================================================================
### README_HARNESS.md
### Author: <Naveen Miriyalu nmiriyal@redhat.com>
### Description: Instructions to run the harness
### =======================================================================

# MLPerf vLLM Harness Usage Guide

This harness supports running vLLM models with MLPerf Loadgen in both offline and server scenarios. The harness for Server and Offline are maintained in two different python code files:
- Offline harness (SUT_VLLM_SingleReplica.py)
  - Offline harness supports running via an LLM API class or sending requests to an OpenAI API endpoint
  - Specifying an --api-server-url would ensure the harness talks to an API endpoint
  - For offline submissions , we use the LLM API class to submit the queries 
- Server harness (SUT_VLLM_SingleReplica_Server.py)
  - Server currently supports only an OpenAI API endpoint
  - Additionally you could set the "target_qps" and "coalesce" controlling loadgen's behavior
- Additionally there are scripts named
  - run_server_submission.sh - Helps to spin up a vllm server and then start the harness and kills the vllm server after the run is completed
  - run_offline_submission.sh - Runs the entire flow to complete a submission using the LLM API class.
- The offline scenario output logs contain the command line used to run within the log
   
> [!TIP]
> These scripts would need slight modifications to specify the right optimizations for the H100 and the L40S GPUs
> Run `SUT_VLLM_SingleReplica*.py -h` if you want a whole list of options the harness supports

## Running the harness for MLPerf Server Scenario 
### 1. On H100 [Specifying the optimized configuration used during final submission]
- A. Spin up the vllm server<br>
`vllm serve --max-model-len 131072 --disable-log-requests --max_seq_len_to_capture 1024 --block-size 16 --gpu-memory-utilization 0.91 --max-num-batched-tokens 1024 --max-num-seq    s 512 --cuda-graph-sizes 4232 --long-prefill-token-threshold 256`
- B. Run the harness for performance <br>
`
python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_PATH} --dataset-path ${DATASET_PATH} \
         --user-conf user.conf  --test-mode performance --target-qps 39.5 --output-log-dir ${MLPERF_OUTPUT_DIR}/ \
          --api-server-url http://localhost:8000 --coalesce >& output.log 
`
- C. Run the harness for accuracy <br>
`
python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_PATH} --dataset-path ${DATASET_PATH} \
         --user-conf user.conf  --test-mode accuracy --target-qps 39.5 --output-log-dir ${MLPERF_OUTPUT_DIR}/ \
          --api-server-url http://localhost:8000 --coalesce >& output.log 
`
- D. Alternatively we could use the `run_server_submission.sh'
  - Use it with the following command line <br> `nohup bash run_server_submission.sh H100 <MODEL_PATH> <DATASET_PATH> <OUTPUT_DIR> 39.5 auto <accuracy|performance|compliance>  --max-model-len 131072 --disable-log-requests   --max_seq_len_to_capture 1024 --block-size 16 --gpu-memory-utilization 0.91 --max-num-batched-tokens  1024 --max-num-seqs 512 --cuda-graph-sizes 4232  --long-prefill-token-threshold 256 & `
  - Additionally in `run_server_submission.sh` do specify the `--coalesce` for the performance run
    ```
    117 if [[ "${CHECK}" == "performance" ]];then
    118 start_vllm performance 0 ${MODEL_DIR} ${CMD}
    119 sleep  50
    120 echo "Run performance"
    121 FILENAME="offline_performance_${GPU}_llama3.18b.log"
    122 python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_DIR} --dataset-path ${DATASET_PATH} \
    123         --user-conf user.conf  --test-mode performance --target-qps ${TARGET_QPS} --output-log-dir ${PERF_DIR}/ \
    124         --api-server-url http://localhost:8000 --coalesce\
    125         >& ${PERF_DIR}/${FILENAME}
    126 echo "Performance run completed"
    127 stop_vllm
    ```

  - The script should set the neccesary environment variables based on the GPU used 
### 2. On L40S [Specifying the optimized configuration used during final submission]
- A. Run `run_server_submission.sh` <br>
 ` bash run_server_submission.sh L40S <MODEL_PATH> <DATASET_PATH>  <OUTPUT_DIR>  9.3  fp8 performance  --kv-cache-dtype fp8 --max-model-len 2668   --gpu-memory-utilization 0.96 --disable-log-requests  `
- B. Alternatively <br>
  - use steps A and B as mentioned for H100. Not that we do not use `--coalesce' for the L40s while running the harness


## Running the harness for MLPerf Offline Scenario 
### 1. On H100 [Specifying the optimized configuration used during final submission]
```
python3 SUT_VLLM_SingleReplica.py --model <model_path> --dataset_path <dataset_path> --user-conf h100_user.conf --test-mode performance --output-log-dir <output_dir> --max-model-len 131072 --max-num-seqs 1024 --kv-cache-dtype fp8 --max-num-batched-tokens 4096 --batch-size 40104
```

### 2. On L40s [Specifying the optimized configuration used during final submission]
 ```
 python3 SUT_VLLM_SingleReplica.py --model <model_path> --dataset_path <dataset_path>  --user-conf user.conf --test-mode performance --output-log-dir <output_dir> --max-model-len 2668 --kv-cache-dtype fp8 --max-model-len 2668 --max-num-seqs 512 --long-prefill-token-threshold 256 --max-num-partial-prefills 1 --max_num_batched_tokens 16384 --cuda-graph-sizes 3000 --max-num-seqs 512 --gpu-memory-utilization 0.95 --batch-size 40104
```
> [!NOTE]
> - To run the harness for compliance add `--audit-conf <path_to_audit>/audit.config` to the harness command line <br>
> - Use the `--test-mode'` option to run the harness for either performance or accuracy <br> 
> - For the offline scenario, you could use 'run_offline_submission.sh' script with necessary changes for the correct optimizations based on the GPU platform 

## Command-Line Options

Options are grouped logically for clarity:

### Model and Data
- `--model-name` : Name or path of the model to load (e.g., HuggingFace repo or local path)
- `--dataset-path` : Path to the processed dataset pickle file
- `--num-samples` : Number of samples/prompts to use


### Performance and Parallelism
- `--batch-size` : Batch size for offline to group data into batches of batch_size
- `--num-gpus` : Number of GPUs (tensor parallel size)

### Loadgen 
- `--test-mode` : Test mode (`performance` or `accuracy`)
- `--api-server-url` : URL of vLLM API server (for API mode) (default: None)
- `--audit-conf`: Audit config for LoadGen settings (default: )
-  `--user-conf`: User config for LoadGen settings (default: user.conf)
-  `--lg-model-name` : {llama3_1-8b,llama3_1-8b-interactive,test-model} Model name for LoadGen (default: llama3_1-8b)

### Logging and Output
- `--log-level` : Logging level (DEBUG, INFO, etc.)
- `--output-log-dir` : Directory for mlperf logs


### Advanced/Debug
- `--enable-profiler` : Enable torch profiler
- `--profiler-dir` : Directory for profiler traces
- `--enable-nvtx` : Enable NVTX profiling
- `--print-histogram` : Print histogram of input token lengths (default: False)
- `--sort-by-length` : Sort queries by input token length (default: False)
- `--sort-by-token-contents` : Sort queries by token contents (default: False)
- `--print-sorted-tokens` : Print sorted tokens
- `--print-timing` : Print timing for each batch 
- `--enable-metrics-csv` : Enable periodic metrics collection (API only) (default: False)
- `--metrics-csv-path` : Path for metrics CSV (API only) (default: metrics.csv)

>[!CAUTION]
> - The Server harness might not support all the listed options.
> - `--target-qps' and '--coalesce` are specific to the server harness
> - Work in progress to unify the harnesses and ensure consistency across  







