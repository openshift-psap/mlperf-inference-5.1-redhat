# ============================================================================
# This file was generated and refactored with the help of AI (OpenAI GPT-4),
# with additional modifications and review by the author: <Naveen Miriyalu nmiriyal@redhat.com>
#
# Disclaimer: This code is provided as-is, without warranty of any kind.
# Please review and test before using in production or submitting to MLPerf.
# ============================================================================
"""
SUT_VLLM_SingleReplica.py
-------------------------
Harness for running vLLM models with MLPerf Loadgen in both offline and server scenarios.
Supports local vLLM, vLLM API, and async server batching with multi-worker support.

This module provides three main SUT (System Under Test) implementations:
1. VLLMSingleSUT - Local vLLM model execution
2. VLLMSingleSUTAPI - Remote vLLM API server communication  
3. VLLMSingleSUTServer - Server scenario with async batching
"""

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
import array
from vllm import AsyncLLMEngine, AsyncEngineArgs
import asyncio


# Import vLLM components with error handling
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print("Please install it using: pip install vllm")
    exit(1)

# Import MLPerf Loadgen with error handling
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    exit(1)

# Import NVTX for profiling (optional)
try:
    import nvtx
except ImportError:
    nvtx = None


# ============================================================================
# MLPerf Loadgen Required Functions
# ============================================================================

def load_samples_to_ram(query_samples):
    """Required by MLPerf Loadgen - samples are pre-loaded in Dataset class"""
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    """Required by MLPerf Loadgen - no action needed for our implementation"""
    del query_samples
    return


# ============================================================================
# Main SUT Classes
# ============================================================================

class VLLMSingleSUT:
    """
    Local vLLM SUT Implementation
    
    This class implements the MLPerf SUT interface for local vLLM model execution.
    It handles batch processing, profiling, and various optimization options.
    Uses per-instance logger for proper logging behavior.
    """
    
    def __init__(self, model_name: str, dataset_path: str, max_model_len: int = None, 
                 gpu_memory_utilization: float = 0.9, max_num_seqs: int = 512, 
                 test_mode: str = "performance", num_gpus: int = 1, 
                 pipeline_parallel_size: int = 0, swap_space: int = 0, 
                 enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", 
                 enable_nvtx: bool = False, print_histogram: bool = False, 
                 sort_by_length: bool = False, sort_by_token_contents: bool = False, 
                 print_sorted_tokens: bool = False, print_timing: bool = False, 
                 max_num_batched_tokens: int = None):
        
        # Initialize per-instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration parameters
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.test_mode = test_mode
        self.num_gpus = num_gpus
        self.pipeline_parallel_size = pipeline_parallel_size
        self.swap_space = swap_space
        self.max_num_batched_tokens = max_num_batched_tokens
        
        # Performance and debugging options
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        
        # Runtime state
        self.profiler = None
        self.batch_counter = 0
        
        # Load dataset and display statistics
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368)
        self.logger.info("Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Dataset statistics - Max Input Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                        max(self.data_object.input_lens), 
                        min(self.data_object.input_lens),
                        len(self.data_object.input_lens))
        
        # Initialize the model
        self._load_model()

    def _load_model(self):
        """Load the vLLM model with specified configuration"""
        # Start NVTX range for model loading if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push("loadmodel")
            
        self.logger.info(f"Loading model '{self.model_name}' with {self.num_gpus} GPU(s)...")
        
        # Create LLM instance with all configuration parameters
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            pipeline_parallel_size=self.pipeline_parallel_size,
            swap_space=self.swap_space,
            disable_log_stats=False,
            max_num_batched_tokens=self.max_num_batched_tokens
        )
        
        self.logger.info("Model loaded successfully.")
        
        # Configure sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            max_tokens=128,   # Maximum output tokens
            min_tokens=1,     # Minimum output tokens
            top_p=1,         # Nucleus sampling parameter
            top_k=1,          # Top-k sampling parameter
            seed=42
        )
        
        # Display model configuration for debugging
        print("-" * 60)
        print("vLLM MODEL CONFIGURATION")
        print("-" * 60)
        try:
            engine_instance = self.llm.llm_engine 
            print("vLLM Config:", engine_instance.vllm_config)
            print("Model Config:", engine_instance.vllm_config.model_config)
            print("Cache Config:", engine_instance.vllm_config.cache_config)
        except Exception as e:
            print(f"Error accessing model configuration: {e}")
        print("-" * 60)

        # End NVTX range for model loading
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()

    def issue_query(self, query_samples: List['lg.QuerySample']):
        """
        Process query samples from MLPerf Loadgen
        
        This is the main entry point called by MLPerf Loadgen. It processes
        queries in batches and returns responses via lg.QuerySamplesComplete().
        """
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        self.logger.info(f"Received {total_samples} queries from Loadgen")
        self.logger.info(f"Processing in {num_batches} batches of size {batch_size}")
        
        batch_times = []
        
        # Initialize profiler for all batches if enabled
        if self.enable_profiler and self.profiler is None:
            self._setup_profiler(num_batches)
        
        # Process each batch
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            
            # Apply sorting options if requested
            batch = self._apply_batch_sorting(batch)
            
            # Print debug information if requested
            if self.print_sorted_tokens or self.logger.isEnabledFor(logging.DEBUG):
                self._print_batch_debug_info(batch_idx, batch)
            
            # Prepare batch data
            prompts_to_process = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index]) 
                                for q in batch]
            original_query_ids = [q.id for q in batch]
            original_query_indexes = [q.index for q in batch]
            
            # Print histogram if requested
            if self.print_histogram:
                self._print_batch_histogram(batch)
            
            # Process the batch
            batch_start = time.time() if self.print_timing else None
            try:
                self._process_single_batch(batch_idx, batch, prompts_to_process, 
                                         original_query_ids, original_query_indexes)
                
                # Record timing if enabled
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'batch_size': len(batch)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                self._handle_batch_error(original_query_ids, batch_start, batch_times, batch_idx, len(batch))
        
        # Cleanup and final statistics
        self._cleanup_profiler()
        if self.print_timing and batch_times:
            self._print_timing_statistics(batch_times)

    def _setup_profiler(self, num_batches):
        """Setup PyTorch profiler for performance analysis"""
        os.makedirs(self.profiler_dir, exist_ok=True)
        trace_file = os.path.join(self.profiler_dir, "vllm_generation_trace.json")
        
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_batches, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.start()
        self.logger.info(f"Profiler started for {num_batches} batches. Trace: {trace_file}")

    def _apply_batch_sorting(self, batch):
        """Apply sorting options to the batch if requested"""
        if self.sort_by_length:
            batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
        elif self.sort_by_token_contents:
            batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
        return batch

    def _print_batch_debug_info(self, batch_idx, batch):
        """Print debug information for the current batch"""
        print(f"Batch {batch_idx} debug info:")
        for i, q in enumerate(batch):
            print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")

    def _print_batch_histogram(self, batch):
        """Print histogram of input lengths and query indexes"""
        input_lens = [len(self.data_object.input_ids[q.index]) for q in batch]
        query_indexes = [q.index for q in batch]
        
        def print_hist_int(data, title, width=50, bins=10):
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
        print_hist_int(query_indexes, "query indexes")
        
        # Check for duplicates
        from collections import Counter
        counter = Counter(sorted(query_indexes))
        duplicates = {k: v for k, v in counter.items() if v > 1}
        if duplicates:
            print("Duplicate query indexes:")
            for idx, freq in duplicates.items():
                print(f"  Query index {idx} repeated {freq} times")
        else:
            print("No duplicate query indexes in this batch.")

    def _process_single_batch(self, batch_idx, batch, prompts_to_process, 
                            original_query_ids, original_query_indexes):
        """Process a single batch through the vLLM model"""
        batch_label = f"batch_{self.batch_counter:04d}_size_{len(batch)}"
        
        # Start NVTX range if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(batch_label)
        
        # Start model profiling if enabled
        if self.enable_profiler:
            self.llm.start_profile()
        
        # Generate responses using vLLM
        with torch.profiler.record_function(batch_label):
            gen_start = time.time() if self.print_timing else None
            outputs = self.llm.generate(prompts_to_process, self.sampling_params)
            gen_end = time.time() if self.print_timing else None
        
        # Stop model profiling
        if self.enable_profiler:
            self.llm.stop_profile()
        
        # End NVTX range
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()
        
        # Process outputs and prepare responses for Loadgen
        responses_to_loadgen = []
        for i in range(len(outputs)):
            output = outputs[i]
            token_ids = output.outputs[0].token_ids
            token_count = len(token_ids)
            query_id = original_query_ids[i]
            query_index = original_query_indexes[i]
            
            # Log output information
            self.logger.info(f"Query ID: {query_id}, Index: {query_index:5d}, Tokens: {token_count}")
            self.logger.debug(f"Token IDs: {token_ids}")
            self.logger.debug(f"Token text: {output.outputs[0].text}")
            
            # Create response based on test mode
            if self.test_mode == "accuracy":
                # For accuracy testing, include actual token data
                token_array = np.array(token_ids, dtype=np.int32)
                token_bytes = token_array.tobytes()
                response_data = token_array.ctypes.data
                response_size = len(token_bytes)
                response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                lg.QuerySamplesComplete([response])
            else:
                # For performance testing, only token count matters
                response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
            
            if self.test_mode == "performance":
                responses_to_loadgen.append(response)
        
        # Send responses to Loadgen
        if responses_to_loadgen and self.test_mode == "performance":
            lg.QuerySamplesComplete(responses_to_loadgen)
        
        self.batch_counter += 1
        
        # Print metrics if timing is enabled
        if self.print_timing:
            self._print_batch_metrics(gen_start, gen_end)

    def _print_batch_metrics(self, gen_start, gen_end):
        """Print detailed metrics for the current batch"""
        if gen_start and gen_end:
            self.logger.info(f"Batch {self.batch_counter} generation time: {gen_end - gen_start:.1f}s")

    def _handle_batch_error(self, original_query_ids, batch_start, batch_times, batch_idx, batch_size):
        """Handle errors during batch processing"""
        # Send error responses to Loadgen
        for query_id in original_query_ids:
            response = lg.QuerySampleResponse(query_id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
        
        self.batch_counter += 1
        
        # Record timing even for failed batches
        if self.print_timing and batch_start:
            batch_end = time.time()
            batch_times.append({
                'batch_idx': batch_idx,
                'start': batch_start,
                'end': batch_end,
                'duration': batch_end - batch_start,
                'batch_size': batch_size,
                'error': True
            })

    def _cleanup_profiler(self):
        """Stop and cleanup the profiler"""
        if self.enable_profiler and self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            self.logger.info(f"Profiler stopped after {self.batch_counter} batches")

    def _print_timing_statistics(self, batch_times):
        """Print comprehensive timing statistics"""
        durations = np.array([bt['duration'] for bt in batch_times])
        
        print("\n" + "="*60)
        print("BATCH TIMING STATISTICS")
        print("="*60)
        print(f"Total batches: {len(batch_times)}")
        print(f"Duration (s): min={durations.min():.1f}, max={durations.max():.1f}")
        print(f"             mean={durations.mean():.1f}, std={durations.std():.1f}")
        print("\nPer-batch details:")
        for bt in batch_times:
            error_flag = " [ERROR]" if bt.get('error', False) else ""
            print(f"  Batch {bt['batch_idx']:3d}: size={bt['batch_size']:4d}, "
                  f"duration={bt['duration']:.1f}s{error_flag}")
        print("="*60)

    def flush_queries(self):
        """MLPerf Loadgen callback - flush any pending queries"""
        self.logger.info("Flush queries called (no action needed for offline scenario)")


class VLLMSingleSUTAPI:
    """
    vLLM API Server SUT Implementation
    
    This class communicates with a remote vLLM API server instead of running
    the model locally. It handles API communication, tokenization/detokenization,
    and optional metrics collection.
    """
    
    def __init__(self, model_name: str, dataset_path: str, api_server_url: str, 
                 max_model_len: int = 2048, test_mode: str = "performance", 
                 enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", 
                 enable_nvtx: bool = False, print_histogram: bool = False, 
                 sort_by_length: bool = False, sort_by_token_contents: bool = False, 
                 print_sorted_tokens: bool = False, print_timing: bool = False, 
                 enable_metrics_csv: bool = False, metrics_csv_path: str = "metrics.csv"):
        
        # Initialize per-instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.api_server_url = api_server_url.rstrip('/')
        self.max_model_len = max_model_len
        self.test_mode = test_mode
        
        # Performance and debugging options
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        
        # Runtime state
        self.batch_counter = 0
        self.server_ready = False
        
        # Metrics collection
        self.enable_metrics_csv = enable_metrics_csv
        self.metrics_csv_path = metrics_csv_path
        self.metrics_thread = None
        self.metrics_stop_event = threading.Event()
        
        # API endpoints
        self.completions_endpoint = f"{self.api_server_url}/v1/completions"
        self.health_endpoint = f"{self.api_server_url}/health"
        self.metrics_endpoint = f"{self.api_server_url}/metrics"
        
        # Load dataset
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368)
        self.logger.info("API Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Dataset statistics - Max Inp Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                        max(self.data_object.input_lens), 
                        min(self.data_object.input_lens),
                        len(self.data_object.input_lens))
        
        # Wait for server readiness and initialize components
        self._wait_for_server_ready()
        self._initialize_tokenizer()
        
        # Start metrics collection if enabled
        if self.enable_metrics_csv:
            self._start_metrics_thread()

    def _wait_for_server_ready(self, timeout: int = 600):
        """Wait for the vLLM API server to become ready"""
        self.logger.info(f"Waiting for API server at {self.api_server_url} (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.health_endpoint, timeout=10)
                if response.status_code == 200:
                    self.logger.info("API server is ready!")
                    self.server_ready = True
                    return
                else:
                    self.logger.warning(f"Health check returned status {response.status_code}")
            except Exception as e:
                self.logger.debug(f"API server not ready: {e}")
            
            time.sleep(1)
        
        raise RuntimeError(f"vLLM API server at {self.api_server_url} did not become ready within {timeout} seconds")
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text encoding/decoding"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.logger.info("Tokenizer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _detokenize_response(self, text_response: str) -> List[int]:
        """Convert text response back to token IDs"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text_response, add_special_tokens=False)
                return tokens
            except Exception as e:
                self.logger.warning(f"Error detokenizing response: {e}")
                return [1, 2, 3]  # Fallback placeholder
        else:
            self.logger.warning("No tokenizer available, using placeholder tokens")
            return [1, 2, 3]  # Fallback placeholder
    
    def issue_query(self, query_samples: List['lg.QuerySample']):
        """Process queries by sending them to the vLLM API server"""
        if not self.server_ready:
            self.logger.error("API server is not ready")
            self._send_error_responses(query_samples)
            return
        
        batch_size = BATCH_SIZE
        total_samples = len(query_samples)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        self.logger.info(f"API processing {total_samples} queries in {num_batches} batches")
        batch_times = []
        
        # Process each batch
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            
            # Apply sorting if requested
            batch = self._apply_batch_sorting(batch)
            
            # Debug information
            if self.print_sorted_tokens or self.logger.isEnabledFor(logging.DEBUG):
                self._print_batch_debug_info(batch_idx, batch)
            
            # Prepare batch data
            original_query_ids = [q.id for q in batch]
            original_query_indexes = [q.index for q in batch]
            
            # Print histogram if requested
            if self.print_histogram:
                self._print_batch_histogram(batch)
            
            # Process the batch via API
            batch_start = time.time() if self.print_timing else None
            try:
                self._process_api_batch(batch_idx, batch, original_query_ids, original_query_indexes)
                
                if self.print_timing:
                    batch_end = time.time()
                    batch_times.append({
                        'batch_idx': batch_idx,
                        'start': batch_start,
                        'end': batch_end,
                        'duration': batch_end - batch_start,
                        'batch_size': len(batch)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing API batch {batch_idx}: {e}")
                self._handle_api_batch_error(original_query_ids, batch_start, batch_times, batch_idx, len(batch))
        
        # Print timing statistics
        if self.print_timing and batch_times:
            self._print_api_timing_statistics(batch_times)

    def _send_error_responses(self, query_samples):
        """Send error responses for all queries"""
        for q_sample in query_samples:
            response = lg.QuerySampleResponse(q_sample.id, 0, 0, 0)
            lg.QuerySamplesComplete([response])

    def _apply_batch_sorting(self, batch):
        """Apply sorting options to the batch"""
        if self.sort_by_length:
            batch = sorted(batch, key=lambda q: len(self.data_object.input_ids[q.index]))
        elif self.sort_by_token_contents:
            batch = sorted(batch, key=lambda q: tuple(self.data_object.input_ids[q.index]))
        return batch

    def _print_batch_debug_info(self, batch_idx, batch):
        """Print debug information for API batch"""
        print(f"API Batch {batch_idx} debug info:")
        for i, q in enumerate(batch):
            print(f"  {i:3d}: idx={q.index}, tokens={self.data_object.input_ids[q.index]}")

    def _print_batch_histogram(self, batch):
        """Print histogram information for API batch"""
        input_lens = [len(self.data_object.input_ids[q.index]) for q in batch]
        self.logger.debug(f"Batch input lengths: min={min(input_lens)}, max={max(input_lens)}")

    def _process_api_batch(self, batch_idx, batch, original_query_ids, original_query_indexes):
        """Process a single batch via the API server"""
        batch_label = f"api_batch_{self.batch_counter:04d}_size_{len(batch)}"
        
        # Start NVTX range if enabled
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(batch_label)
        
        with torch.profiler.record_function(batch_label):
            gen_start = time.time() if self.print_timing else None
            
            # Convert token IDs to text prompts
            text_prompts = self._prepare_text_prompts(batch)
            
            # Prepare API request
            api_payload = {
                "model": self.model_name,
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
        
        # End NVTX range
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()
        
        # Process API responses
        self._process_api_responses(choices, original_query_ids, original_query_indexes)
        
        self.batch_counter += 1
        
        # Log timing if enabled
        if self.print_timing and gen_start and gen_end:
            self.logger.info(f"API batch {batch_idx} processing time: {gen_end - gen_start:.1f}s")

    def _prepare_text_prompts(self, batch):
        """Convert token IDs to text prompts for API"""
        text_prompts = []
        for q_sample in batch:
            if self.tokenizer:
                try:
                    text_prompt = self.tokenizer.decode(
                        self.data_object.input_ids[q_sample.index], 
                        skip_special_tokens=True
                    )
                    text_prompts.append(text_prompt)
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens for query {q_sample.id}: {e}")
                    # Fallback to token ID string
                    text_prompts.append(" ".join([str(t) for t in self.data_object.input_ids[q_sample.index]]))
            else:
                # No tokenizer available, use token IDs as string
                text_prompts.append(" ".join([str(t) for t in self.data_object.input_ids[q_sample.index]]))
        return text_prompts

    def _process_api_responses(self, choices, original_query_ids, original_query_indexes):
        """Process API responses and send to Loadgen"""
        responses_to_loadgen = []
        
        for i, choice in enumerate(choices):
            query_id = original_query_ids[i]
            query_index = original_query_indexes[i]
            
            # Extract text response from API
            text_response = choice.get("text", "")
            
            # Convert back to token IDs
            token_ids = self._detokenize_response(text_response)
            token_count = len(token_ids)
            
            # Debug logging
            self.logger.info(f"API Query ID: {query_id}, Index: {query_index}, Tokens: {token_count}")
            self.logger.debug(f"API Token IDs: {token_ids}")
            self.logger.debug(f"API Text Response: {text_response}")
            
            # Create response based on test mode
            if self.test_mode == "accuracy":
                token_array = np.array(token_ids, dtype=np.int32)
                token_bytes = token_array.tobytes()
                response_data = token_array.ctypes.data
                response_size = len(token_bytes)
                response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                lg.QuerySamplesComplete([response])
            else:
                response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
            
            if self.test_mode == "performance":
                responses_to_loadgen.append(response)
        
        # Send responses to Loadgen
        if responses_to_loadgen and self.test_mode == "performance":
            lg.QuerySamplesComplete(responses_to_loadgen)

    def _handle_api_batch_error(self, original_query_ids, batch_start, batch_times, batch_idx, batch_size):
        """Handle errors in API batch processing"""
        for query_id in original_query_ids:
            response = lg.QuerySampleResponse(query_id, 0, 0, 0)
            lg.QuerySamplesComplete([response])
        
        self.batch_counter += 1
        
        if self.print_timing and batch_start:
            batch_end = time.time()
            batch_times.append({
                'batch_idx': batch_idx,
                'start': batch_start,
                'end': batch_end,
                'duration': batch_end - batch_start,
                'batch_size': batch_size,
                'error': True
            })

    def _print_api_timing_statistics(self, batch_times):
        """Print timing statistics for API processing"""
        durations = np.array([bt['duration'] for bt in batch_times])
        
        print("\n" + "="*60)
        print("API BATCH TIMING STATISTICS")
        print("="*60)
        print(f"Total batches: {len(batch_times)}")
        print(f"Duration (s): min={durations.min():.1f}, max={durations.max():.1f}")
        print(f"             mean={durations.mean():.1f}, std={durations.std():.1f}")
        print("="*60)

    def flush_queries(self):
        """MLPerf Loadgen callback for flushing queries"""
        self.logger.info("API SUT flush queries called")

    def _start_metrics_thread(self):
        """Start background thread for metrics collection"""
        def metrics_worker():
            self.logger.info(f"Starting metrics collection to {self.metrics_csv_path}")
            with open(self.metrics_csv_path, mode='w', newline='') as csvfile:
                writer = None
                while not self.metrics_stop_event.is_set():
                    try:
                        response = requests.get(self.metrics_endpoint, timeout=10)
                        if response.status_code == 200:
                            metrics_data = response.text
                            timestamp = datetime.now().isoformat()
                            
                            # Parse Prometheus format metrics
                            lines = [l for l in metrics_data.splitlines() if l and not l.startswith('#')]
                            metrics_dict = {l.split()[0]: l.split()[1] for l in lines if len(l.split()) == 2}
                            metrics_dict['timestamp'] = timestamp
                            
                            if writer is None:
                                fieldnames = list(metrics_dict.keys())
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                            
                            writer.writerow(metrics_dict)
                            csvfile.flush()
                        else:
                            self.logger.warning(f"Metrics endpoint returned status {response.status_code}")
                    except Exception as e:
                        self.logger.warning(f"Error collecting metrics: {e}")
                    
                    self.metrics_stop_event.wait(1)  # 1 second interval
            
            self.logger.info("Metrics collection stopped")
        
        self.metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
        self.metrics_thread.start()

    def stop_metrics_thread(self):
        """Stop the metrics collection thread"""
        if self.enable_metrics_csv and self.metrics_thread is not None:
            self.metrics_stop_event.set()
            self.metrics_thread.join()
            self.logger.info("Metrics collection thread stopped")


class VLLMSingleSUTServer:
    """
    vLLM Server Scenario SUT Implementation
    
    This class implements the MLPerf Server scenario using AsyncLLMEngine.
    It supports multi-worker batching and async query processing for server workloads.
    Uses per-instance logger for proper logging behavior.
    """
    
    def __init__(self, model_name: str, dataset_path: str, max_model_len: int = None, 
                 gpu_memory_utilization: float = 0.9, max_num_seqs: int = 512, 
                 test_mode: str = "performance", num_gpus: int = 1, 
                 pipeline_parallel_size: int = 0, swap_space: int = 0, 
                 enable_profiler: bool = False, profiler_dir: str = "./torch_profiler_logs", 
                 enable_nvtx: bool = False, print_histogram: bool = False, 
                 sort_by_length: bool = False, sort_by_token_contents: bool = False, 
                 print_sorted_tokens: bool = False, print_timing: bool = False, 
                 max_num_batched_tokens: int = None, num_workers: int = 1, 
                 batch_size: int = 1):
        
        # Initialize per-instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configuration parameters
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.test_mode = test_mode
        self.num_gpus = num_gpus
        self.pipeline_parallel_size = pipeline_parallel_size
        self.swap_space = swap_space
        self.max_num_batched_tokens = max_num_batched_tokens
        
        # Performance and debugging options
        self.enable_profiler = enable_profiler
        self.profiler_dir = profiler_dir
        self.enable_nvtx = enable_nvtx
        self.print_histogram = print_histogram
        self.sort_by_length = sort_by_length
        self.sort_by_token_contents = sort_by_token_contents
        self.print_sorted_tokens = print_sorted_tokens
        self.print_timing = print_timing
        
        # Server-specific configuration
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.query_queue = queue.Queue()
        self.worker_threads = [None] * self.num_workers
        self.request_id = 0
        
        # Runtime state
        self.profiler = None
        self.batch_counter = 0
        
        # Load dataset and display statistics
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=13368)
        self.logger.info("Server Dataset loaded: %d samples", len(self.data_object.input_ids))
        self.logger.info("Server will use %d workers with batch size %d", self.num_workers, self.batch_size)
        
        # Initialize the model and start workers
        self._load_model()
        self._start_workers()

    def _load_model(self):
        """Load the AsyncLLMEngine for server scenario"""
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push("server_loadmodel")
            
        self.logger.info(f"Loading AsyncLLMEngine for '{self.model_name}' with {self.num_gpus} GPU(s)...")
        
        # Create AsyncLLMEngine arguments
        self.engine_args = AsyncEngineArgs(
            self.model_name,
            tensor_parallel_size=self.num_gpus,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            pipeline_parallel_size=self.pipeline_parallel_size,
            swap_space=self.swap_space,
            max_num_batched_tokens=self.max_num_batched_tokens
        )
        
        # Create the async engine
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        
        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            max_tokens=128,   # Maximum output tokens
            min_tokens=1,     # Minimum output tokens
            top_p=1,         # Nucleus sampling parameter
            top_k=1,          # Top-k sampling parameter
            seed=42          # Top-k sampling parameter
        )
        
        self.logger.info("AsyncLLMEngine loaded successfully.")
        
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()

    def _start_workers(self):
        """Start worker threads for processing queries"""
        self.logger.info(f"Starting {self.num_workers} worker threads")
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries, name=f"Worker-{j}")
            worker.daemon = True
            worker.start()
            self.worker_threads[j] = worker

    async def stream_output(self, batch, results_generator, original_query_ids, original_query_indexes):
        """
        Stream output tokens and handle first token completion
        
        This method processes the async generator from the AsyncLLMEngine,
        handles first token completion, and sends final responses.
        """
        first = True
        outputs = None
        
        async for request_output in results_generator:
            outputs = request_output.outputs
            if first:
                # Send first token completion for server scenario
                for i, output in enumerate(outputs):
                    token_ids = output.token_ids
                    if token_ids:  # Only if we have tokens
                        response_data = array.array("B", np.array(token_ids, np.int32).tobytes())
                        bi = response_data.buffer_info()
                        response = [lg.QuerySampleResponse(original_query_ids[i], bi[0], bi[1])]
                        lg.FirstTokenComplete(response)
                first = False
        
        # After streaming, send final QuerySamplesComplete for all
        if outputs:
            for i, output in enumerate(outputs):
                token_ids = output.token_ids
                n_tokens = len(token_ids)
                query_id = original_query_ids[i]
                
                self.logger.debug(f"Server Query ID: {query_id}, Final Tokens: {n_tokens}")
                
                if self.test_mode == "accuracy":
                    response_array = array.array("B", np.array(token_ids, np.int32).tobytes())
                    bi = response_array.buffer_info()
                    response = [lg.QuerySampleResponse(query_id, bi[0], bi[1], n_tokens)]
                else:
                    response = [lg.QuerySampleResponse(query_id, 0, 0, n_tokens)]
                
                lg.QuerySamplesComplete(response)

    def process_queries(self):
        """
        Worker thread main loop for processing queries
        
        Each worker dequeues up to batch_size queries and processes them
        as a batch using the AsyncLLMEngine.
        """
        while True:
            batch = []
            try:
                # Block until at least one query is available
                qitem = self.query_queue.get()
                if qitem is None:  # Shutdown signal
                    break
                batch.append(qitem)
                
                # Try to get up to batch_size-1 more queries without blocking
                for _ in range(self.batch_size - 1):
                    try:
                        qitem = self.query_queue.get_nowait()
                        if qitem is None:  # Shutdown signal
                            break
                        batch.append(qitem)
                    except queue.Empty:
                        break
                
                if not batch:
                    continue
                
                # Process the batch
                self._process_worker_batch(batch)
                
            except Exception as e:
                self.logger.error(f"Error in worker thread: {e}")
                # Send error responses for this batch
                for q in batch:
                    response = lg.QuerySampleResponse(q.id, 0, 0, 0)
                    lg.QuerySamplesComplete([response])

    def _process_worker_batch(self, batch):
        """Process a batch of queries in a worker thread"""
        # Prepare batch input
        prompts_to_process = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index]) 
                            for q in batch]
        original_query_ids = [q.id for q in batch]
        original_query_indexes = [q.index for q in batch]
        
        self.logger.debug(f"Worker processing batch of {len(batch)} queries")
        
        # Generate outputs using AsyncLLMEngine
        try:
            results_generator = self.model.generate(
                prompt=prompts_to_process, 
                sampling_params=self.sampling_params, 
                request_id=str(self.request_id)
            )
            self.request_id += 1
            
            # Stream outputs and handle responses
            asyncio.run(self.stream_output(batch, results_generator, original_query_ids, original_query_indexes))
            
        except Exception as e:
            self.logger.error(f"Error generating responses: {e}")
            # Send error responses
            for query_id in original_query_ids:
                response = lg.QuerySampleResponse(query_id, 0, 0, 0)
                lg.QuerySamplesComplete([response])

    def issue_query(self, query_samples):
        """
        Issue queries to the server SUT
        
        For server scenario, each query is enqueued individually for
        processing by worker threads.
        """
        self.logger.debug(f"Enqueuing {len(query_samples)} queries for server processing")
        
        for q in query_samples:
            self.query_queue.put(q)

    def flush_queries(self):
        """MLPerf Loadgen callback - flush any pending queries"""
        self.logger.info("Server SUT flush queries called")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Print comprehensive system information
    import sys
    print("=" * 80)
    print("MLPERF vLLM HARNESS - SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Executable: {sys.executable}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print installed packages for debugging
    print("Key Python packages:")
    pkgs = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], 
                  key=lambda x: x[0].lower())
    for name, version in pkgs:
        if any(keyword in name.lower() for keyword in ['torch', 'vllm', 'mlperf', 'transformers','tokenizers','cuda']):
            print(f"  {name:<30} {version}")
    print("=" * 80)

    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = "16"

    # Set TORCH_CUDA_ARCH_LIST based on device properties
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        arch_str = f"{props.major}.{props.minor}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_str
        print(f"Set TORCH_CUDA_ARCH_LIST to {arch_str}")
    else:
        print("CUDA not available. Not setting TORCH_CUDA_ARCH_LIST")

    # ========================================================================
    # Command Line Argument Parsing
    # ========================================================================
    
    parser = argparse.ArgumentParser(
        description="MLPerf vLLM Harness - Run vLLM models with MLPerf Loadgen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and Data Configuration
    model_group = parser.add_argument_group('Model and Data')
    model_group.add_argument("--model-name", type=str, 
                           default="HuggingFaceH4/tiny-random-LlamaForCausalLM", 
                           help="The name of the LLM model to load")
    model_group.add_argument("--dataset-path", type=str, default=None, 
                           help="Path to the processed dataset pickle file")
    model_group.add_argument("--num-samples", type=int, default=13368, 
                           help="Number of samples for the test")
    
    # Performance Configuration
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument("--max-model-len", type=int, default=131072, 
                          help="Maximum sequence length for the model")
    perf_group.add_argument("--max-num-seqs", type=int, default=512, 
                          help="Maximum sequences processed simultaneously")
    perf_group.add_argument("--gpu-mem-util", type=float, default=0.9, 
                          help="GPU memory utilization factor (0.0 to 1.0)")
    perf_group.add_argument("--batch-size", type=int, default=32, 
                          help="Batch size for processing")
    perf_group.add_argument("--max-num-batched-tokens", type=int, default=None, 
                          help="Maximum number of batched tokens for vLLM")
    perf_group.add_argument("--num-workers", type=int, default=1, 
                          help="Number of worker threads for server scenario")
    
    # Scenario and Testing
    scenario_group = parser.add_argument_group('Scenario and Testing')
    scenario_group.add_argument("--scenario", type=str, default="Offline", 
                              choices=["Offline", "Server"], 
                              help="MLPerf scenario")
    scenario_group.add_argument("--test-mode", type=str, default="performance", 
                              choices=["performance", "accuracy"], 
                              help="Test mode")
    
    # Hardware Configuration
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument("--num-gpus", type=int, default=1, 
                        help="Number of GPUs (tensor_parallel_size)")
    hw_group.add_argument("--pipeline-parallel-size", type=int, default=1, 
                        help="Pipeline parallel size")
    hw_group.add_argument("--swap-space", type=int, default=4, 
                        help="Swap space parameter")
    
    # Logging and Output
    log_group = parser.add_argument_group('Logging and Output')
    log_group.add_argument("--log-level", type=str, default="INFO", 
                         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                         help="Logging level")
    log_group.add_argument("--output-log-dir", type=str, default="./", 
                         help="Directory for log output")
    
    # MLPerf Loadgen Configuration
    lg_group = parser.add_argument_group('MLPerf Loadgen')
    lg_group.add_argument("--user-conf", type=str, default="user.conf", 
                        help="User config for LoadGen settings")
    lg_group.add_argument("--lg-model-name", type=str, default="llama3_1-8b", 
                        choices=["llama3_1-8b", "llama3_1-8b-interactive", "test-model"], 
                        help="Model name for LoadGen")
    
    # Profiling and Analysis
    prof_group = parser.add_argument_group('Profiling and Analysis')
    prof_group.add_argument("--enable-profiler", action="store_true", 
                          help="Enable torch profiler")
    prof_group.add_argument("--profiler-dir", type=str, default="./torch_profiler_logs", 
                          help="Directory for profiler traces")
    prof_group.add_argument("--enable-nvtx", action="store_true", 
                          help="Enable NVTX profiling")
    prof_group.add_argument("--print-timing", action="store_true", 
                          help="Print timing statistics")
    
    # Data Analysis and Debugging
    debug_group = parser.add_argument_group('Data Analysis and Debugging')
    debug_group.add_argument("--print-histogram", action="store_true", 
                           help="Print histogram of input lengths")
    debug_group.add_argument("--sort-by-length", action="store_true", 
                           help="Sort queries by input token length")
    debug_group.add_argument("--sort-by-token-contents", action="store_true", 
                           help="Sort queries by token contents")
    debug_group.add_argument("--print-sorted-tokens", action="store_true", 
                           help="Print input token lists after sorting")
    
    # API Server Options
    api_group = parser.add_argument_group('API Server')
    api_group.add_argument("--api-server-url", type=str, default=None, 
                         help="URL of vLLM API server")
    api_group.add_argument("--enable-metrics-csv", action="store_true", 
                         help="Enable metrics collection (API only)")
    api_group.add_argument("--metrics-csv-path", type=str, default="metrics.csv", 
                         help="Path for metrics CSV file")
    
    args = parser.parse_args()

    # ========================================================================
    # Environment Setup
    # ========================================================================
    
    # Set profiler directory if enabled
    if args.enable_profiler:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profiler_dir
    os.environ["VLLM_NO_USAGE_STATS"] = "0"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Extract configuration variables
    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len
    MAX_NUM_SEQS = args.max_num_seqs
    GPU_MEM_UTIL = args.gpu_mem_util
    BATCH_SIZE = args.batch_size
    TEST_MODE = args.test_mode
    SCENARIO = args.scenario
    NUM_GPUS = args.num_gpus
    PIPELINE_PARALLEL_SIZE = args.pipeline_parallel_size
    SWAP_SPACE = args.swap_space
    MAX_NUM_BATCHED_TOKENS = args.max_num_batched_tokens
    NUM_WORKERS = args.num_workers

    # Validation
    if DATASET_PATH is None:
        logging.error("Error: --dataset-path is required")
        exit(1)

    if NUM_SAMPLES <= 0:
        logging.error("Error: --num-samples must be at least 1")
        exit(1)

    # ========================================================================
    # SUT Selection and Initialization
    # ========================================================================
    
    logging.info("=" * 50)
    logging.info("INITIALIZING MLPerf vLLM HARNESS")
    logging.info("=" * 50)

    sut = None
    try:
        if args.api_server_url:
            # Use API server implementation
            logging.info(f"Using vLLM API server at: {args.api_server_url}")
            sut = VLLMSingleSUTAPI(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                api_server_url=args.api_server_url,
                max_model_len=MAX_MODEL_LEN,
                test_mode=TEST_MODE,
                enable_profiler=args.enable_profiler,
                profiler_dir=args.profiler_dir,
                enable_nvtx=args.enable_nvtx,
                print_histogram=args.print_histogram,
                sort_by_length=args.sort_by_length,
                sort_by_token_contents=args.sort_by_token_contents,
                print_sorted_tokens=args.print_sorted_tokens,
                print_timing=args.print_timing,
                enable_metrics_csv=args.enable_metrics_csv,
                metrics_csv_path=args.metrics_csv_path
            )
        elif SCENARIO == "Server":
            # Use server scenario implementation
            logging.info("Using local vLLM model for Server scenario")
            sut = VLLMSingleSUTServer(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                max_model_len=MAX_MODEL_LEN,
                gpu_memory_utilization=GPU_MEM_UTIL,
                max_num_seqs=MAX_NUM_SEQS,
                test_mode=TEST_MODE,
                num_gpus=NUM_GPUS,
                pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
                swap_space=SWAP_SPACE,
                enable_profiler=args.enable_profiler,
                profiler_dir=args.profiler_dir,
                enable_nvtx=args.enable_nvtx,
                print_histogram=args.print_histogram,
                sort_by_length=args.sort_by_length,
                sort_by_token_contents=args.sort_by_token_contents,
                print_sorted_tokens=args.print_sorted_tokens,
                print_timing=args.print_timing,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                num_workers=NUM_WORKERS,
                batch_size=BATCH_SIZE
            )
        else:
            # Use local model for offline scenario
            logging.info("Using local vLLM model for Offline scenario")
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
                enable_profiler=args.enable_profiler,
                profiler_dir=args.profiler_dir,
                enable_nvtx=args.enable_nvtx,
                print_histogram=args.print_histogram,
                sort_by_length=args.sort_by_length,
                sort_by_token_contents=args.sort_by_token_contents,
                print_sorted_tokens=args.print_sorted_tokens,
                print_timing=args.print_timing,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
            )

        # ====================================================================
        # MLPerf Loadgen Configuration and Execution
        # ====================================================================
        
        # Configure test settings
        settings = lg.TestSettings()
        if SCENARIO == "Server":
            settings.scenario = lg.TestScenario.Server
        else:
            settings.scenario = lg.TestScenario.Offline
            
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
            
        settings.use_token_latencies = True
        settings.FromConfig(args.user_conf, args.lg_model_name, SCENARIO, 1)

        # Configure logging settings
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False

        # Create Query Sample Library
        qsl = lg.ConstructQSL(13368, NUM_SAMPLES, load_samples_to_ram, unload_samples_from_ram)
        
        # Create SUT for Loadgen
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

        # Log test configuration
        logging.info("=" * 50)
        logging.info("STARTING MLPerf TEST")
        logging.info("=" * 50)
        logging.info(f"Model: {MODEL_NAME}")
        logging.info(f"Scenario: {SCENARIO}")
        logging.info(f"Test Mode: {TEST_MODE}")
        logging.info(f"Samples: {NUM_SAMPLES}")
        logging.info(f"Batch Size: {BATCH_SIZE}")
        if SCENARIO == "Server":
            logging.info(f"Server Workers: {NUM_WORKERS}")
        if args.enable_profiler:
            logging.info(f"Profiling enabled - traces in {args.profiler_dir}")
        if args.enable_nvtx:
            logging.info("NVTX profiling enabled")

        # Run the test
        lg.StartTestWithLogSettings(SUTToTest, qsl, settings, log_settings)

        # Test completion
        logging.info("=" * 50)
        logging.info("MLPerf TEST COMPLETED SUCCESSFULLY")
        logging.info("=" * 50)

        # Cleanup
        if args.api_server_url and args.enable_metrics_csv and hasattr(sut, 'stop_metrics_thread'):
            sut.stop_metrics_thread()

    except Exception as e:
        logging.critical(f"Critical error in main program: {e}", exc_info=True)
        exit(1) 
