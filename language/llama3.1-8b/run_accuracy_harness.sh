CHECKPOINT_PATH="${CHECKPOINT_PATH:meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:cnn_eval.json}"

python3 SUT_VLLM_SingleReplica.py --scenario Offline \
        --model-name ${CHECKPOINT_PATH} \
        --dataset-path ${DATASET_PATH} \
        --user-conf user.conf \
        --batch-size 13368 \
        --test-mode accuracy \
        --output-log-dir offline_accuracy_loadgen_logs | tee offline_accuracy_log.log 

python3 evaluation.py \
        --mlperf-accuracy-file offline_accuracy_loadgen_logs/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32
