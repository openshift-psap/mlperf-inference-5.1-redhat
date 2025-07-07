import os
import time
import logging
import argparse
import numpy as np
from typing import List
from dataset import Dataset
from vllm import TokensPrompt
import sys
import torch
import pkg_resources
from datetime import datetime

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print("Please install it using: pip install vllm")
    exit(1)

try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    exit(1)

try:
    import nvtx
except ImportError:
    nvtx = None

def load_samples_to_ram(query_samples):
    del query_samples
    return

def unload_samples_from_ram(query_samples):
    del query_samples
    return

class VLLMSingleSUT:
    def __init__(self, model_name: str, dataset_path: str, max_model_len: int = None, gpu_memory_utilization: float = 0.9, max_num_seqs: int = 512, test_mode: str = "performance", num_gpus: int = 1, pipeline_parallel_size: int = 0, swap_space: int = 0, enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", enable_nvtx: bool = False):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.test_mode = test_mode
        self.num_gpus = num_gpus
        self.pipeline_parallel_size = pipeline_parallel_size
        self.swap_space = swap_space
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.profiler = None
        self.batch_counter = 0
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=24576, device="cpu")
        logging.info("Datatset = %d", len(self.data_object.input_ids))
        logging.info("Datatset Max = %d", max(self.data_object.input_lens))
        logging.info("Datatset Min = %d", min(self.data_object.input_lens))
        logging.info("Datatset Len = %d", len(self.data_object.input_lens))
        self._load_model()

    def _load_model(self):
        #if self.enable_nvtx and nvtx:
        #    nvtx.push_range("loadmodel")
        logging.info(f"Loading model '{self.model_name}' on single GPU...")
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            pipeline_parallel_size=self.pipeline_parallel_size,
            swap_space=self.swap_space
        )
        logging.info("Model loaded successfully.")
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
        )
        
        print("--------------START CONFIG---------------------------\n")
        try:
            engine_instance = self.llm.llm_engine 
            print(engine_instance.vllm_config) 
            print(engine_instance.vllm_config.model_config) 
            print(engine_instance.vllm_config.cache_config) 
        except Exception as e:
            print(f"  An unexpected error occurred while dumping config: {e}")
        print("--------------END   CONFIG---------------------------\n")

        #if self.enable_nvtx and nvtx:
        #    nvtx.pop_range()

    def issue_query(self, query_samples: List['lg.QuerySample']):
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        logging.info(f"SUT issue_query: Received {len(query_samples)} queries from Loadgen. Batch size: {batch_size}. Number of batches: {num_batches}.")
        
        # Initialize profiler once for all batches if enabled
        if self.enable_profiler and self.profiler is None:
            os.makedirs(self.profiler_dir, exist_ok=True)
            trace_file = os.path.join(self.profiler_dir, "vllm_generation_trace.json")
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=num_batches,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
            logging.info(f"Started torch profiler for all {num_batches} batches. Trace will be saved to {trace_file}")
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            prompts_to_process = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q_sample.index]) for q_sample in batch]
            original_query_ids = [q_sample.id for q_sample in batch]
            
            try:
                #if self.enable_nvtx and nvtx:
                #    nvtx.push_range("llmgenerate")
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_push("testing")
                
                # Use PyTorch record function to mark this batch
                batch_label = f"batch_{self.batch_counter:04d}_size_{len(batch)}"
                # Only profile if enabled
                if self.enable_profiler:
                    self.llm.start_profile()
                with torch.profiler.record_function(batch_label):
                    outputs = self.llm.generate(prompts_to_process, self.sampling_params)
                
                if self.enable_profiler:
                    self.llm.stop_profile()
                #if self.enable_nvtx and nvtx:
                #    nvtx.pop_range()
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_pop()
                
                responses_to_loadgen = []
                for i, output in enumerate(outputs):
                    token_ids = output.outputs[0].token_ids
                    token_count = len(token_ids)
                    query_id = original_query_ids[i]
                    if self.test_mode == "accuracy":
                        token_array = np.array(token_ids, dtype=np.int32)
                        token_bytes = token_array.tobytes()
                        response_data = token_array.ctypes.data
                        response_size = len(token_bytes)
                        response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                    else:
                        response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
                    responses_to_loadgen.append(response)
                if responses_to_loadgen:
                    lg.QuerySamplesComplete(responses_to_loadgen)
                
                self.batch_counter += 1
                
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                for query_id in original_query_ids:
                    response = lg.QuerySampleResponse(query_id, 0, 0, 0)
                    lg.QuerySamplesComplete([response])
                
                self.batch_counter += 1
        
        # Stop profiler after all batches are processed
        if self.enable_profiler and self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            logging.info(f"Stopped torch profiler after processing {self.batch_counter} batches")

    def flush_queries(self):
        logging.info("SUT flush_queries: Flushing (no specific action for offline in this demo).")

if __name__ == "__main__":
    # Print command line and executable information
    import sys
    print("="*80)
    print("COMMAND LINE AND EXECUTABLE INFORMATION")
    print("="*80)
    print(f"Executable: {sys.executable}")
    print(f"Command line: {' '.join(sys.argv)}")
    print("="*80)
    # Print date and time
    now = datetime.now()
    print(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    # Print installed packages
    print("Installed Python packages:")
    pkgs = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], key=lambda x: x[0].lower())
    for name, version in pkgs:
        print(f"  {name:<30} {version}")
    print("="*80)
    print()

    # Set OMP_NUM_THREADS
    os.environ['OMP_NUM_THREADS'] = "16"

    # Set TORCH_CUDA_ARCH_LIST based on device properties
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        arch_str = f"{props.major}.{props.minor}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_str
        print(f"Set TORCH_CUDA_ARCH_LIST to {arch_str}")
    else:
        print("CUDA is not available, not setting TORCH_CUDA_ARCH_LIST")

    parser = argparse.ArgumentParser(
        description="Run vLLM generation with MLPerf Loadgen in offline scenario (single replica).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/tiny-random-LlamaForCausalLM", help="The name of the LLM model to load.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the processed dataset pickle file containing tokenized inputs")
    parser.add_argument("--num_samples", type=int, default=24576, help="Number of samples (prompts) Loadgen will issue for the offline test.")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--max_num_seqs", type=int, default=512, help="Maximum number of sequences that can be processed simultaneously by vLLM")
    parser.add_argument("--gpu_mem_util", type=float, default=0.9, help="GPU memory utilization factor (0.0 to 1.0) for vLLM model loading")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--batch_size", type=int, default=3072, help="Batch size for the single worker process.")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--lg_model_name", type=str, default="llama2-70b", choices=["llama2-70b", "llama2-70b-interactive","test-model"], help="Model name(specified in llm server)")
    parser.add_argument("--output-log-dir", type=str, default="./", help="Where logs are saved")
    parser.add_argument("--test-mode", type=str, default="performance", choices=["performance", "accuracy"], help="Test mode: 'performance' for performance testing, 'accuracy' for accuracy testing with raw bytes logging")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (tensor_parallel_size)")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size (default 1)")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space parameter for vLLM")
    parser.add_argument("--enable-profiler", action="store_true", help="Enable torch profiler to profile LLM generate calls with batch labeling")
    parser.add_argument("--profiler-dir", type=str, default="./torch_profiler_logs", help="Directory to save torch profiler traces")
    parser.add_argument("--enable-nvtx", action="store_true", help="Enable NVTX profiling for GPU timeline analysis")
    args = parser.parse_args()

    # Set profiler directory environment variable
    os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profiler_dir

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.ERROR),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len
    MAX_NUM_SEQS = args.max_num_seqs
    GPU_MEM_UTIL = args.gpu_mem_util
    BATCH_SIZE = args.batch_size
    TEST_MODE = args.test_mode
    NUM_GPUS = args.num_gpus
    PIPELINE_PARALLEL_SIZE = args.pipeline_parallel_size
    SWAP_SPACE = args.swap_space
    ENABLE_PROFILER = args.enable_profiler
    PROFILER_DIR = args.profiler_dir
    ENABLE_NVTX = args.enable_nvtx

    if DATASET_PATH is None:
        logging.error("Error: --dataset_path is required.")
        exit(1)

    if NUM_SAMPLES <= 0:
        logging.error("Error: Number of samples (--num_samples) must be at least 1.")
        exit(1)

    logging.info("-" * 50)

    sut = None
    try:
        sut = VLLMSingleSUT(
            model_name=MODEL_NAME,
            dataset_path=DATASET_PATH,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=MAX_NUM_SEQS,
            test_mode=TEST_MODE,
            num_gpus=NUM_GPUS,
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            swap_space=SWAP_SPACE,
            enable_profiler=ENABLE_PROFILER,
            profiler_dir=PROFILER_DIR,
            enable_nvtx=ENABLE_NVTX
        )
        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
        settings.use_token_latencies = True
        settings.FromConfig(args.user_conf, args.lg_model_name, "Offline")
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False
        qsl = lg.ConstructQSL(
            24576,
            NUM_SAMPLES,
            load_samples_to_ram,
            unload_samples_from_ram
        )
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)
        logging.info(f"MLPerf Loadgen: Starting test with {NUM_SAMPLES} samples in Offline mode...")
        logging.info(f"Model: {MODEL_NAME}, Test Mode: {TEST_MODE}")
        if ENABLE_PROFILER:
            logging.info(f"Torch profiler is enabled - all batches will be profiled in a single trace file in {PROFILER_DIR}")
        if ENABLE_NVTX:
            logging.info("NVTX profiling is enabled - GPU timeline markers will be added")
        lg.StartTestWithLogSettings(SUTToTest, qsl, settings, log_settings)
        logging.info("\nMLPerf Loadgen test finished.")
        logging.info("Main: Program finished.")
        logging.info("Run Completed!")
    except Exception as e:
        logging.critical(f"\nMain program encountered an error: {e}") 