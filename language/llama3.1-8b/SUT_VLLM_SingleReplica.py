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
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import csv



#os.environ["VLLM_TORCH_PROFILER_DIR"] = "./PROFILESFRESH"

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
    def __init__(self, model_name: str, dataset_path: str, max_model_len: int = None, gpu_memory_utilization: float = 0.9, max_num_seqs: int = 512, test_mode: str = "performance", num_gpus: int = 1, pipeline_parallel_size: int = 0, swap_space: int = 0, enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", enable_nvtx: bool = False, print_histogram: bool = False, sort_by_length: bool = False, sort_by_token_contents: bool = False, print_sorted_tokens: bool = False, print_timing: bool = False):
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
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        self.profiler = None
        self.batch_counter = 0
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368)
        logging.info("Dataset Max          = %d", max(self.data_object.input_lens))
        logging.info("Dataset Min          = %d", min(self.data_object.input_lens))
        logging.info("Dataset TotalSamples = %d", len(self.data_object.input_lens))
        self._load_model()

    def _load_model(self):
        if self.enable_nvtx :
            torch.cuda.nvtx.range_push("loadmodel")
        logging.info(f"Loading model '{self.model_name}' on single GPU...")
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            pipeline_parallel_size=self.pipeline_parallel_size,
            swap_space=self.swap_space,
            disable_log_stats=False
        )
        logging.info("Model loaded successfully.")
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
            min_tokens=1,
            top_p=1,
            top_k=1
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

        if self.enable_nvtx :
            torch.cuda.nvtx.range_pop()

    def issue_query(self, query_samples: List['lg.QuerySample']):
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        logging.info(f"SUT issue_query: Received {len(query_samples)} queries from Loadgen. Batch size: {batch_size}. Number of batches: {num_batches}.")
        batch_times = []
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
            # Optionally sort by input length
            if self.sort_by_length:
                batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
            # Optionally sort by token contents
            if self.sort_by_token_contents:
                batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
            # Optionally print sorted tokens
            if self.print_sorted_tokens or logging.getLogger().isEnabledFor(logging.DEBUG):
                print(f"Batch {batch_idx} sorted tokens:")
                for i, q in enumerate(batch):
                    print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")
            prompts_to_process = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q_sample.index]) for q_sample in batch]
            original_query_ids = [q_sample.id for q_sample in batch]
            original_query_indexes = [q_sample.index for q_sample in batch]
            # Optionally print histogram
            if self.print_histogram:
                input_lens = [len(self.data_object.input_ids[q_sample.index]) for q_sample in batch]
                query_indexes = [q_sample.index for q_sample in batch]
                def print_hist_int(data, title, width=50, bins=10):
                    import numpy as np
                    data = np.array(data, dtype=int)
                    min_val, max_val = int(np.min(data)), int(np.max(data))
                    if min_val == max_val:
                        bins = 1
                    else:
                        bins = min(bins, max_val - min_val + 1)
                    hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val+1))
                    max_count = max(hist)
                    print(f"Histogram of {title} (integer bins):")
                    for i in range(len(hist)):
                        left = int(bin_edges[i])
                        right = int(bin_edges[i+1]) - 1
                        bar = '#' * int(width * hist[i] / max_count) if max_count > 0 else ''
                        print(f"  {left:6d} - {right:6d}: {bar} ({hist[i]})")
                print_hist_int(input_lens, "input token lengths")
                # Query index histogram and duplicate report
                print_hist_int(query_indexes, "query indexes")
                # Duplicate/repetition report
                from collections import Counter
                sorted_qidx = sorted(query_indexes)
                counter = Counter(sorted_qidx)
                duplicates = {k: v for k, v in counter.items() if v > 1}
                if duplicates:
                    print("Duplicate/repeated query indexes:")
                    for idx, freq in duplicates.items():
                        print(f"  Query index {idx} repeated {freq} times")
                else:
                    print("No duplicate/repeated query indexes in this batch.")
            # Optionally print timing
            import time
            batch_start = time.time() if self.print_timing else None
            try:
                batch_label = f"batch_{self.batch_counter:04d}_size_{len(batch)}"
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_push(batch_label)
                # Only profile if enabled
                if self.enable_profiler:
                    self.llm.start_profile()
                with torch.profiler.record_function(batch_label):
                    gen_start = time.time() if self.print_timing else None
                    outputs = self.llm.generate(prompts_to_process, self.sampling_params)
                    gen_end = time.time() if self.print_timing else None
                if self.enable_profiler:
                    self.llm.stop_profile()
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_pop()
                responses_to_loadgen = []
                for i, output in enumerate(outputs):
                    token_ids = output.outputs[0].token_ids
                    token_count = len(token_ids)
                    query_id = original_query_ids[i]
                    query_index = original_query_indexes[i]
                    
                    # Detailed debug logging for output tokens
                    logging.debug(f"Query ID: {query_id}, Query Index: {query_index}, Output Tokens: {token_count}")
                    logging.debug(f"Token IDs: {token_ids}")
                    
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
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'llm_generate': (gen_end - gen_start) if gen_start is not None and gen_end is not None else None,
                        'batch_size': len(batch)
                    })
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                for query_id in original_query_ids:
                    response = lg.QuerySampleResponse(query_id, 0, 0, 0)
                    lg.QuerySamplesComplete([response])
                self.batch_counter += 1
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'llm_generate': None,
                        'batch_size': len(batch)
                    })
        # Stop profiler after all batches are processed
        if self.enable_profiler and self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            logging.info(f"Stopped torch profiler after processing {self.batch_counter} batches")
        # Print timing stats if enabled
        if self.print_timing and batch_times:
            import numpy as np
            durations = np.array([bt['duration'] for bt in batch_times])
            gen_durations = np.array([bt['llm_generate'] for bt in batch_times if bt['llm_generate'] is not None])
            print("\nBatch timing statistics:")
            print(f"  Batches: {len(batch_times)}")
            print(f"  Duration (s): min={durations.min():.4f}, max={durations.max():.4f}, mean={durations.mean():.4f}, std={durations.std():.4f}")
            if len(gen_durations) > 0:
                print(f"  LLM generate (s): min={gen_durations.min():.4f}, max={gen_durations.max():.4f}, mean={gen_durations.mean():.4f}, std={gen_durations.std():.4f}")
            print("  Per-batch details:")
            for bt in batch_times:
                print(f"    Batch {bt['batch_idx']:3d}: size={bt['batch_size']:4d}, duration={bt['duration']:.4f}s, llm_generate={bt['llm_generate'] if bt['llm_generate'] is not None else 'N/A'}")
            for metric in self.llm.get_metrics():
                        if isinstance(metric, Gauge):
                            print(f"{metric.name} (gauge) = {metric.value}")
                        elif isinstance(metric, Counter):
                            print(f"{metric.name} (counter) = {metric.value}")
                        elif isinstance(metric, Vector):
                            print(f"{metric.name} (vector) = {metric.values}")
                        elif isinstance(metric, Histogram):
                            print(f"{metric.name} (histogram)")
                            print(f"    sum = {metric.sum}")
                            print(f"    count = {metric.count}")
                            for bucket_le, value in metric.buckets.items():
                                print(f"    {bucket_le} = {value}")

    def flush_queries(self):
        logging.info("SUT flush_queries: Flushing (no specific action for offline in this demo).")

class VLLMSingleSUTAPI:
    """Completely separate class for handling vLLM API server communication"""
    
    def __init__(self, model_name: str, dataset_path: str, api_server_url: str, max_model_len: int = None, test_mode: str = "performance", enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", enable_nvtx: bool = False, print_histogram: bool = False, sort_by_length: bool = False, sort_by_token_contents: bool = False, print_sorted_tokens: bool = False, print_timing: bool = False, enable_metrics_csv: bool = False, metrics_csv_path: str = "metrics.csv"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.api_server_url = api_server_url.rstrip('/')
        self.max_model_len = max_model_len
        self.test_mode = test_mode
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        self.batch_counter = 0
        self.server_ready = False
        self.enable_metrics_csv = enable_metrics_csv
        self.metrics_csv_path = metrics_csv_path
        self.metrics_thread = None
        self.metrics_stop_event = threading.Event()
        
        # API endpoints
        self.completions_endpoint = f"{self.api_server_url}/v1/completions"
        self.health_endpoint = f"{self.api_server_url}/health"
        self.metrics_endpoint = f"{self.api_server_url}/metrics"
        
        # Load dataset
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368 )
        logging.info("Dataset = %d", len(self.data_object.input_ids))
        logging.info("Dataset Max = %d", max(self.data_object.input_lens))
        logging.info("Dataset Min = %d", min(self.data_object.input_lens))
        logging.info("Dataset Len = %d", len(self.data_object.input_lens))
        
        # Wait for server to be ready
        self._wait_for_server_ready()
        
        # Initialize tokenizer for detokenization (if available)
        self._initialize_tokenizer()

        # Start metrics thread if enabled
        if self.enable_metrics_csv:
            self._start_metrics_thread()

    def _wait_for_server_ready(self, timeout: int = 600):
        """Wait for the vLLM API server to be ready with timeout"""
        logging.info(f"Waiting for vLLM API server at {self.api_server_url} to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_endpoint, timeout=10)
                if response.status_code == 200:
                    logging.info(f"vLLM API server at {self.api_server_url} is ready!")
                    self.server_ready = True
                    return
                else:
                    logging.warning(f"API server health check returned status {response.status_code}")
            except Exception as e:
                logging.debug(f"API server not ready yet: {e}")
            
            time.sleep(5)  # Wait 5 seconds before next check
        
        raise RuntimeError(f"vLLM API server at {self.api_server_url} did not become ready within {timeout} seconds")
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for detokenization if available"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logging.info("Tokenizer initialized for detokenization")
        except Exception as e:
            logging.warning(f"Could not initialize tokenizer for detokenization: {e}")
            self.tokenizer = None
    
    def _detokenize_response(self, text_response: str) -> List[int]:
        """Detokenize text response back to token IDs"""
        if self.tokenizer:
            try:
                # Tokenize the response text to get token IDs
                tokens = self.tokenizer.encode(text_response, add_special_tokens=False)
                return tokens
            except Exception as e:
                logging.warning(f"Error detokenizing response: {e}")
                # Fallback: return a simple token representation
                return [1, 2, 3]  # Placeholder tokens
        else:
            # No tokenizer available, return placeholder
            logging.warning("No tokenizer available for detokenization, using placeholder tokens")
            return [1, 2, 3]  # Placeholder tokens
    
    def issue_query(self, query_samples: List['lg.QuerySample']):
        """Handle queries by sending them to the vLLM API server"""
        if not self.server_ready:
            logging.error("API server is not ready")
            # Send error responses
            for q_sample in query_samples:
                response = lg.QuerySampleResponse(q_sample.id, 0, 0, 0)
                lg.QuerySamplesComplete([response])
            return
        
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        logging.info(f"API SUT issue_query: Received {len(query_samples)} queries from Loadgen. Batch size: {batch_size}. Number of batches: {num_batches}.")
        batch_times = []
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            
            # Optionally sort by input length
            if self.sort_by_length:
                batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
            # Optionally sort by token contents
            if self.sort_by_token_contents:
                batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
            
            # Optionally print sorted tokens
            if self.print_sorted_tokens or logging.getLogger().isEnabledFor(logging.DEBUG):
                print(f"Batch {batch_idx} sorted tokens:")
                for i, q in enumerate(batch):
                    print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")
            
            original_query_ids = [q_sample.id for q_sample in batch]
            original_query_indexes = [q_sample.index for q_sample in batch]
            
            # Optionally print histogram
            if self.print_histogram:
                input_lens = [len(self.data_object.input_ids[q_sample.index]) for q_sample in batch]
                query_indexes = [q_sample.index for q_sample in batch]
                def print_hist_int(data, title, width=50, bins=10):
                    import numpy as np
                    data = np.array(data, dtype=int)
                    min_val, max_val = int(np.min(data)), int(np.max(data))
                    if min_val == max_val:
                        bins = 1
                    else:
                        bins = min(bins, max_val - min_val + 1)
                    hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val+1))
                    max_count = max(hist)
                    print(f"Histogram of {title} (integer bins):")
                    for i in range(len(hist)):
                        left = int(bin_edges[i])
                        right = int(bin_edges[i+1]) - 1
                        bar = '#' * int(width * hist[i] / max_count) if max_count > 0 else ''
                        print(f"  {left:6d} - {right:6d}: {bar} ({hist[i]})")
                print_hist_int(input_lens, "input token lengths")
                # Query index histogram and duplicate report
                print_hist_int(query_indexes, "query indexes")
                # Duplicate/repetition report
                from collections import Counter
                sorted_qidx = sorted(query_indexes)
                counter = Counter(sorted_qidx)
                duplicates = {k: v for k, v in counter.items() if v > 1}
                if duplicates:
                    print("Duplicate/repeated query indexes:")
                    for idx, freq in duplicates.items():
                        print(f"  Query index {idx} repeated {freq} times")
                else:
                    print("No duplicate/repeated query indexes in this batch.")
            
            # Optionally print timing
            batch_start = time.time() if self.print_timing else None
            try:
                batch_label = f"api_batch_{self.batch_counter:04d}_size_{len(batch)}"
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_push(batch_label)
                
                with torch.profiler.record_function(batch_label):
                    gen_start = time.time() if self.print_timing else None
                    
                    # Convert token IDs to text prompts for API server
                    text_prompts = []
                    for q_sample in batch:
                        # Convert token IDs to text using tokenizer if available
                        if self.tokenizer:
                            try:
                                text_prompt = self.tokenizer.decode(self.data_object.input_ids[q_sample.index], skip_special_tokens=True)
                                text_prompts.append(text_prompt)
                            except Exception as e:
                                logging.warning(f"Error decoding tokens for query {q_sample.id}: {e}")
                                # Fallback: use token IDs as string
                                text_prompts.append(" ".join([str(t) for t in self.data_object.input_ids[q_sample.index]]))
                        else:
                            # No tokenizer, use token IDs as string
                            text_prompts.append(" ".join([str(t) for t in self.data_object.input_ids[q_sample.index]]))
                    
                    # Prepare API request
                    api_payload = {
                        "model": self.model_name,  # Use default model
                        "prompt": text_prompts,
                        "max_tokens": 128,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "stream": False
                    }
                    
                    # Send request to API server
                    response = requests.post(self.completions_endpoint, json=api_payload)
                    if response.status_code != 200:
                        raise RuntimeError(f"API server returned status {response.status_code}: {response.text}")
                    
                    api_result = response.json()
                    choices = api_result.get("choices", [])
                    
                    gen_end = time.time() if self.print_timing else None
                
                if self.enable_nvtx:
                    torch.cuda.nvtx.range_pop()
                
                # Process API responses
                detok_start = time.time() if self.print_timing else None
                responses_to_loadgen = []
                for i, choice in enumerate(choices):
                    query_id = original_query_ids[i]
                    query_index = original_query_indexes[i]
                    
                    # Extract text response from API
                    text_response = choice.get("text", "")
                    
                    # Detokenize response
                    if self.print_timing:
                        print(f"[Timing] Detokenizing response for query_id={query_id}")
                    token_ids = self._detokenize_response(text_response)
                    token_count = len(token_ids)
                    
                    # Detailed debug logging for output tokens
                    logging.info(f"API Query ID: {query_id}, Query Index: {query_index}, Output Tokens: {token_count}")
                    logging.debug(f"API Token IDs: {token_ids}")
                    logging.debug(f"API Text Response: {text_response}")
                    
                    if self.test_mode == "accuracy":
                        token_array = np.array(token_ids, dtype=np.int32)
                        token_bytes = token_array.tobytes()
                        response_data = token_array.ctypes.data
                        response_size = len(token_bytes)
                        response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                    else:
                        response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
                    responses_to_loadgen.append(response)
                if self.print_timing and detok_start is not None:
                    detok_end = time.time()
                    print(f"[Timing] Total detokenization time for batch {batch_idx}: {detok_end - detok_start:.6f} seconds")
                
                if responses_to_loadgen:
                    lg.QuerySamplesComplete(responses_to_loadgen)
                
                self.batch_counter += 1
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'api_generate': (gen_end - gen_start) if gen_start is not None and gen_end is not None else None,
                        'batch_size': len(batch)
                    })
                
            except Exception as e:
                logging.error(f"Error processing API batch: {e}")
                for query_id in original_query_ids:
                    response = lg.QuerySampleResponse(query_id, 0, 0, 0)
                    lg.QuerySamplesComplete([response])
                self.batch_counter += 1
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'api_generate': None,
                        'batch_size': len(batch)
                    })
        
        # Print timing stats if enabled
        if self.print_timing and batch_times:
            import numpy as np
            durations = np.array([bt['duration'] for bt in batch_times])
            gen_durations = np.array([bt['api_generate'] for bt in batch_times if bt['api_generate'] is not None])
            print("\nAPI Batch timing statistics:")
            print(f"  Batches: {len(batch_times)}")
            print(f"  Duration (s): min={durations.min():.4f}, max={durations.max():.4f}, mean={durations.mean():.4f}, std={durations.std():.4f}")
            if len(gen_durations) > 0:
                print(f"  API generate (s): min={gen_durations.min():.4f}, max={gen_durations.max():.4f}, mean={gen_durations.mean():.4f}, std={gen_durations.std():.4f}")
            print("  Per-batch details:")
            for bt in batch_times:
                print(f"    Batch {bt['batch_idx']:3d}: size={bt['batch_size']:4d}, duration={bt['duration']:.4f}s, api_generate={bt['api_generate'] if bt['api_generate'] is not None else 'N/A'}")

    def flush_queries(self):
        logging.info("API SUT flush_queries: Flushing (no specific action for offline in this demo).")

    def _start_metrics_thread(self):
        def metrics_worker():
            logging.info(f"Starting metrics collection thread, writing to {self.metrics_csv_path}")
            with open(self.metrics_csv_path, mode='w', newline='') as csvfile:
                writer = None
                while not self.metrics_stop_event.is_set():
                    try:
                        response = requests.get(self.metrics_endpoint, timeout=10)
                        if response.status_code == 200:
                            metrics_data = response.text
                            timestamp = datetime.now().isoformat()
                            # For Prometheus format, parse lines as key value pairs
                            lines = [l for l in metrics_data.splitlines() if l and not l.startswith('#')]
                            metrics_dict = {l.split()[0]: l.split()[1] for l in lines if len(l.split()) == 2}
                            metrics_dict['timestamp'] = timestamp
                            if writer is None:
                                # Write header
                                fieldnames = list(metrics_dict.keys())
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                            writer.writerow(metrics_dict)
                            csvfile.flush()
                        else:
                            logging.warning(f"Metrics endpoint returned status {response.status_code}")
                    except Exception as e:
                        logging.warning(f"Error collecting metrics: {e}")
                    self.metrics_stop_event.wait(1)  # 1 second interval
            logging.info("Metrics collection thread stopped.")
        self.metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
        self.metrics_thread.start()

    def stop_metrics_thread(self):
        if self.enable_metrics_csv and self.metrics_thread is not None:
            self.metrics_stop_event.set()
            self.metrics_thread.join()
            logging.info("Metrics thread joined.")

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
    parser.add_argument("--num_samples", type=int, default=13368, help="Number of samples (prompts) Loadgen will issue for the offline test.")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum sequence length for the model.")
    parser.add_argument("--max_num_seqs", type=int, default=512, help="Maximum number of sequences that can be processed simultaneously by vLLM")
    parser.add_argument("--gpu_mem_util", type=float, default=0.9, help="GPU memory utilization factor (0.0 to 1.0) for vLLM model loading")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the single worker process.")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--lg_model_name", type=str, default="llama3_1-8b", choices=["llama3_1-8b", "llama3_1-8b-interactive","test-model"], help="Model name(specified in llm server)")
    parser.add_argument("--output-log-dir", type=str, default="./", help="Where logs are saved")
    parser.add_argument("--test-mode", type=str, default="performance", choices=["performance", "accuracy"], help="Test mode: 'performance' for performance testing, 'accuracy' for accuracy testing with raw bytes logging")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (tensor_parallel_size)")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size (default 1)")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space parameter for vLLM")
    parser.add_argument("--enable-profiler", action="store_true", help="Enable torch profiler to profile LLM generate calls with batch labeling")
    parser.add_argument("--profiler-dir", type=str, default="./torch_profiler_logs", help="Directory to save torch profiler traces")
    parser.add_argument("--enable-nvtx", action="store_true", help="Enable NVTX profiling for GPU timeline analysis")
    parser.add_argument("--print-histogram", action="store_true", help="Print histogram of input lengths and query indexes for each batch")
    parser.add_argument("--sort-by-length", action="store_true", help="Sort queries in each batch by input token length before passing to LLM")
    parser.add_argument("--sort-by-token-contents", action="store_true", help="Sort queries in each batch by the contents of the input token list (lexicographically)")
    parser.add_argument("--print-sorted-tokens", action="store_true", help="Print the input token lists for each batch after sorting")
    parser.add_argument("--print-timing", action="store_true", help="Print timing statistics for each batch and overall timing stats")
    parser.add_argument("--api-server-url", type=str, default=None, help="URL of vLLM API server to use instead of local model")
    parser.add_argument("--enable-metrics-csv", action="store_true", help="Enable periodic metrics collection from /metrics endpoint (SUTAPI only)")
    parser.add_argument("--metrics-csv-path", type=str, default="metrics.csv", help="Path to CSV file for metrics logging (SUTAPI only)")
    args = parser.parse_args()

    # Set profiler directory environment variable only if profiler is enabled
    if args.enable_profiler:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profiler_dir
    os.environ["VLLM_NO_USAGE_STATS"] = "0"

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
    PRINT_HISTOGRAM = args.print_histogram
    SORT_BY_LENGTH = args.sort_by_length
    SORT_BY_TOKEN_CONTENTS = args.sort_by_token_contents
    PRINT_SORTED_TOKENS = args.print_sorted_tokens
    PRINT_TIMING = args.print_timing

    if DATASET_PATH is None:
        logging.error("Error: --dataset_path is required.")
        exit(1)

    if NUM_SAMPLES <= 0:
        logging.error("Error: Number of samples (--num_samples) must be at least 1.")
        exit(1)

    logging.info("-" * 50)

    sut = None
    try:
        # Choose between local model and API server based on command line argument
        if args.api_server_url:
            # Use API server
            logging.info(f"Using vLLM API server at: {args.api_server_url}")
            sut = VLLMSingleSUTAPI(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                api_server_url=args.api_server_url,
                max_model_len=MAX_MODEL_LEN,
                test_mode=TEST_MODE,
                enable_profiler=ENABLE_PROFILER,
                profiler_dir=PROFILER_DIR,
                enable_nvtx=ENABLE_NVTX,
                print_histogram=PRINT_HISTOGRAM,
                sort_by_length=SORT_BY_LENGTH,
                sort_by_token_contents=SORT_BY_TOKEN_CONTENTS,
                print_sorted_tokens=PRINT_SORTED_TOKENS,
                print_timing=PRINT_TIMING,
                enable_metrics_csv=args.enable_metrics_csv,
                metrics_csv_path=args.metrics_csv_path
            )
        else:
            # Use local model
            logging.info("Using local vLLM model")
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
                enable_nvtx=ENABLE_NVTX,
                print_histogram=PRINT_HISTOGRAM,
                sort_by_length=SORT_BY_LENGTH,
                sort_by_token_contents=SORT_BY_TOKEN_CONTENTS,
                print_sorted_tokens=PRINT_SORTED_TOKENS,
                print_timing=PRINT_TIMING
            )
        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
        settings.use_token_latencies = True
        settings.FromConfig(args.user_conf, args.lg_model_name, "Offline",1)
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False
        qsl = lg.ConstructQSL(
            13368,
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
        # Stop metrics thread if SUTAPI and enabled
        if args.api_server_url and args.enable_metrics_csv and hasattr(sut, 'stop_metrics_thread'):
            sut.stop_metrics_thread()
    except Exception as e:
        logging.critical(f"\nMain program encountered an error: {e}") 
