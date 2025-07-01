
# HARNESS:
- MLPerf Offline Scenario Support: Designed for efficient benchmarking of vLLM in MLPerf's offline inference scenario.
- Scalable VLLM Deployment: Allows specification of both the number of vLLM replicas and the total GPUs to be used.
- Automatic GPU Distribution: Ensures GPUs are automatically and efficiently distributed across each vLLM replica.
- Guaranteed Model Readiness: Incorporates a critical synchronization step that waits for all vLLM model replicas to be fully loaded and ready before commencing interaction with the Loadgen.
- Comprehensive Configuration: All available configuration options and parameters can be found by running the harness with the -h option.
- Most of the code is AI generated 

## Performance Only Run Command Line

```python
python3 SUT_VLLM.py --model_name <model_path> --dataset_path <dataset_path> --lg_model_name llama2-70b --user-conf user.conf  --batch_size <batch_size>  --num_samples <num_performance_samples_to_run> --num_replicas <num_replicas> --num_gpus <num_gpus> --output-log-dir <output_dir> --gpu_mem_util 0.8 --test-mode performance >& output.log 
```
> - *output_dir* - This is where the mlperf results shall be stored . This directory is not created automatically
> - *num_replicas* - determines how many vLLM instances are going to be created
> - *num_gpus* - total gpus in the system

***Please look for additional command line options via -h***
  


## Obtaining Accuracy 

Run the test.  This shall produce mlperf_log_accuracy.json
```python
python3 SUT_VLLM.py --model_name ${BASEDIR}/models/Llama2-70b-fp8/ --dataset_path ${BASEDIR}/datasets/Llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl  --user-conf user.conf  --batch_size 24576   --num_replicas 1 --num_gpus 4 --output-log-dir <mlperf_output> --gpu_mem_util 0.8 --test-mode accuracy >& output.log 
```

I created a separate environment to install required packages for accuracy check
```python
uv pip install transformers==4.31.0 nlt
k==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0
```

```python
python3 evaluate-accuracy.py --checkpoint-path  ${BASEDIR}/models/Llama2-70b-fp8/ --mlperf-accuracy-file   mlperf_log_accuracy.json --dataset-file  ${BASEDIR}/datasets/Llama270b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl  --dtype int32
```
