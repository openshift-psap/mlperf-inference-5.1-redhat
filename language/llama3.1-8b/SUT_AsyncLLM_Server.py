#!/usr/bin/env python3

import asyncio
import logging
import argparse
import time
import threading
import queue
import array
import numpy as np
from typing import List, Optional, Union
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
from dataset import Dataset

# Import MLPerf Loadgen with error handling
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAsyncLLMEngine:
    """
    A simple AsyncLLMEngine wrapper that supports both text prompts and token IDs,
    with focus on batched streaming generation.
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: Optional[int] = None):
        """
        Initialize the AsyncLLMEngine.
        
        Args:
            model_path: Path to the model (local path or HuggingFace model name)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_model_len: Maximum sequence length for the model
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.engine = None
        self.sampling_params = None
        self.tokenizer = None
        
        # Sample prompts for testing
        self.sample_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?"
        ]
        
        # Will be populated after tokenizer is loaded
        self.sample_token_ids = []
    
    async def initialize(self):
        """Initialize the AsyncLLMEngine and tokenizer with the specified configuration."""
        logger.info(f"Initializing AsyncLLMEngine with model: {self.model_path}")
        
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize sample prompts
        logger.info("Tokenizing sample prompts...")
        self.sample_token_ids = []
        for prompt in self.sample_prompts:
            token_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            self.sample_token_ids.append(token_ids)
            logger.info(f"Prompt: '{prompt[:30]}...' -> {len(token_ids)} tokens")
        
        # Create engine arguments
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            trust_remote_code=True,  # Often needed for newer models
        )
        
        # Create the async engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,    # Some randomness for interesting responses
            max_tokens=128,     # Maximum output tokens
            min_tokens=1,       # Minimum output tokens
            top_p=0.9,         # Nucleus sampling
            top_k=50,          # Top-k sampling
        )
        
        logger.info("AsyncLLMEngine initialized successfully")
    
    def text_to_token_ids(self, text: str) -> List[int]:
        """Convert text to token IDs using the loaded tokenizer."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call initialize() first.")
        return self.tokenizer.encode(text, return_tensors="pt")[0].tolist()
    
    def token_ids_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text using the loaded tokenizer."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call initialize() first.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # # COMMENTED OUT: Single prompt generation
    # async def generate_single(self, prompt: Union[str, List[int]], request_id: Optional[str] = None) -> str:
    #     """
    #     Generate a response for a single prompt (text or token IDs).
    #     
    #     Args:
    #         prompt: The input prompt (text string or list of token IDs)
    #         request_id: Optional request ID for tracking
    #         
    #     Returns:
    #         Generated text response
    #     """
    #     if not self.engine:
    #         raise RuntimeError("Engine not initialized. Call initialize() first.")
    #     
    #     # Convert to appropriate format
    #     if isinstance(prompt, str):
    #         logger.info(f"Generating response for text prompt: {prompt[:50]}...")
    #         vllm_prompt = prompt
    #     else:
    #         logger.info(f"Generating response for token IDs: {len(prompt)} tokens")
    #         vllm_prompt = TokensPrompt(prompt_token_ids=prompt)
    #     
    #     # Generate response
    #     results_generator = self.engine.generate(
    #         prompt=vllm_prompt,
    #         sampling_params=self.sampling_params,
    #         request_id=request_id or "single_gen"
    #     )
    #     
    #     # Collect the generated text
    #     generated_text = ""
    #     async for request_output in results_generator:
    #         if request_output.outputs:
    #             generated_text = request_output.outputs[0].text
    #     
    #     return generated_text
    
    # # COMMENTED OUT: Single prompt streaming generation
    # async def generate_single_streamed(self, prompt: Union[str, List[int]], request_id: Optional[str] = None) -> str:
    #     """
    #     Generate a response for a single prompt with visible streaming output.
    #     
    #     Args:
    #         prompt: The input prompt (text string or list of token IDs)
    #         request_id: Optional request ID for tracking
    #         
    #     Returns:
    #         Generated text response
    #     """
    #     if not self.engine:
    #         raise RuntimeError("Engine not initialized. Call initialize() first.")
    #     
    #     # Convert to appropriate format and display info
    #     if isinstance(prompt, str):
    #         logger.info(f"Streaming response for text prompt: {prompt[:50]}...")
    #         print(f"\n🚀 Streaming Response:")
    #         print(f"📝 Prompt: {prompt}")
    #         vllm_prompt = prompt
    #     else:
    #         prompt_text = self.token_ids_to_text(prompt)
    #         logger.info(f"Streaming response for token IDs: {len(prompt)} tokens")
    #         print(f"\n🚀 Streaming Response:")
    #         print(f"📝 Prompt ({len(prompt)} tokens): {prompt_text}")
    #         vllm_prompt = TokensPrompt(prompt_token_ids=prompt)
    #     
    #     print(f"💭 Response: ", end="", flush=True)
    #     
    #     # Generate response
    #     results_generator = self.engine.generate(
    #         prompt=vllm_prompt,
    #         sampling_params=self.sampling_params,
    #         request_id=request_id or "stream_gen"
    #     )
    #     
    #     # Stream and capture the generated text
    #     generated_text = ""
    #     previous_text = ""
    #     token_count = 0
    #     start_time = time.time()
    #     first_token_time = None
    #     
    #     async for request_output in results_generator:
    #         if request_output.outputs:
    #             current_text = request_output.outputs[0].text
    #             generated_text = current_text
    #             
    #             # Print only the new tokens (incremental output)
    #             new_tokens = current_text[len(previous_text):]
    #             if new_tokens:
    #                 # Record first token latency
    #                 if first_token_time is None:
    #                     first_token_time = time.time() - start_time
    #                 
    #                 print(new_tokens, end="", flush=True)
    #                 previous_text = current_text
    #                 token_count += len(new_tokens.split())
    #                 
    #                 # Small delay to make streaming more visible
    #                 await asyncio.sleep(0.02)
    #     
    #     end_time = time.time()
    #     total_time = end_time - start_time
    #     tokens_per_second = token_count / total_time if total_time > 0 else 0
    #     
    #     print(f"\n✅ Streaming complete!")
    #     print(f"📊 Stats: ~{token_count} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    #     if first_token_time:
    #         print(f"⚡ First token latency: {first_token_time:.3f}s\n")
    #     else:
    #         print()
    #     
    #     return generated_text
    
    # # COMMENTED OUT: Regular batch generation
    # async def generate_batch(self, prompts: List[Union[str, List[int]]]) -> List[str]:
    #     """
    #     Generate responses for a batch of prompts (text or token IDs).
    #     
    #     Args:
    #         prompts: List of input prompts (text strings or lists of token IDs)
    #         
    #     Returns:
    #         List of generated text responses
    #     """
    #     if not self.engine:
    #         raise RuntimeError("Engine not initialized. Call initialize() first.")
    #     
    #     logger.info(f"Generating responses for {len(prompts)} prompts")
    #     
    #     # Generate responses for each prompt individually and collect them
    #     responses = []
    #     for i, prompt in enumerate(prompts):
    #         response = await self.generate_single(prompt, f"batch_{i}")
    #         responses.append(response)
    #     
    #     return responses
    
    async def generate_batch_streamed(self, prompts: List[Union[str, List[int]]]) -> List[str]:
        """
        Generate responses for a batch of prompts with concurrent streaming output.
        Supports both text prompts and token IDs.
        
        Args:
            prompts: List of input prompts (text strings or lists of token IDs)
            
        Returns:
            List of generated text responses
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        logger.info(f"Streaming responses for {len(prompts)} prompts concurrently")
        
        # Color codes for different prompts (ANSI escape codes)
        colors = [
            '\033[91m',  # Red
            '\033[92m',  # Green  
            '\033[93m',  # Yellow
            '\033[94m',  # Blue
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
        ]
        reset_color = '\033[0m'
        
        print(f"\n🚀 Streaming {len(prompts)} prompts concurrently:")
        for i, prompt in enumerate(prompts):
            color = colors[i % len(colors)]
            if isinstance(prompt, str):
                print(f"{color}📝 Prompt {i+1} (text): {prompt}{reset_color}")
            else:
                prompt_text = self.token_ids_to_text(prompt)
                print(f"{color}📝 Prompt {i+1} ({len(prompt)} tokens): {prompt_text}{reset_color}")
        print("\n" + "="*80)
        
        # Create async tasks for concurrent generation
        async def stream_single_prompt(prompt_idx: int, prompt: Union[str, List[int]]) -> str:
            color = colors[prompt_idx % len(colors)]
            prefix = f"[P{prompt_idx+1}]"
            
            # Convert to appropriate vLLM format
            if isinstance(prompt, str):
                vllm_prompt = prompt
            else:
                vllm_prompt = TokensPrompt(prompt_token_ids=prompt)
            
            # Generate response
            results_generator = self.engine.generate(
                prompt=vllm_prompt,
                sampling_params=self.sampling_params,
                request_id=f"batch_stream_{prompt_idx}"
            )
            
            # Stream and capture the generated text
            generated_text = ""
            previous_text = ""
            token_count = 0
            start_time = time.time()
            first_token_time = None
            
            async for request_output in results_generator:
                if request_output.outputs:
                    current_text = request_output.outputs[0].text
                    generated_text = current_text
                    
                    # Print only the new tokens (incremental output)
                    new_tokens = current_text[len(previous_text):]
                    if new_tokens:
                        # Record first token latency
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        
                        # Print with color and prompt identifier
                        print(f"{color}{prefix} {new_tokens}{reset_color}", flush=True)
                        previous_text = current_text
                        token_count += len(new_tokens.split())
                        
                        # Small delay to make streaming more visible
                        await asyncio.sleep(0.03)
            
            end_time = time.time()
            total_time = end_time - start_time
            tokens_per_second = token_count / total_time if total_time > 0 else 0
            
            print(f"{color}{prefix} ✅ Complete! ~{token_count} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s){reset_color}")
            if first_token_time:
                print(f"{color}{prefix} ⚡ First token: {first_token_time:.3f}s{reset_color}")
            
            return generated_text
        
        # Start all prompts concurrently
        start_time = time.time()
        tasks = [stream_single_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        responses = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        print(f"\n🎉 All {len(prompts)} prompts completed in {total_time:.2f}s!")
        print("="*80 + "\n")
        
        return responses
    
    async def run_sample_generation(self, use_token_ids: bool = True):
        """Run generation on the sample prompts to demonstrate functionality."""
        logger.info("Running sample generation with predefined prompts")
        
        if use_token_ids:
            logger.info("=== Batch Streaming Demo (Using Token IDs) ===")
            # Use first 3 prompts as token IDs
            batch_prompts = self.sample_token_ids[:3]
        else:
            logger.info("=== Batch Streaming Demo (Using Text) ===")
            # Use first 3 prompts as text
            batch_prompts = self.sample_prompts[:3]
        
        await self.generate_batch_streamed(batch_prompts)


class MLPerfAsyncSUT:
    """
    MLPerf System Under Test (SUT) implementation for AsyncLLMEngine
    with Server scenario support and proper LoadGen integration.
    """
    
    def __init__(self, 
                 model_path: str,
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 1024,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 batch_timeout_ms: float = 10.0,
                 dataset_path: Optional[str] = None,
                 test_mode: str = "performance"):
        """
        Initialize MLPerf SUT with AsyncLLMEngine.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            batch_size: Maximum batch size for processing
            num_workers: Number of worker threads
            batch_timeout_ms: Timeout in milliseconds to wait for batch completion
            dataset_path: Path to dataset file (if using external dataset)
            test_mode: Either "accuracy" or "performance" mode
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_timeout_ms = batch_timeout_ms
        self.dataset_path = dataset_path
        self.test_mode = test_mode
        
        # AsyncLLMEngine components
        self.llm_engine = None
        self.tokenizer = None
        
        # MLPerf components
        self.query_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_threads = []
        self.async_processor_thread = None
        self.is_running = False
        self.request_id = 0
        self.engine_ready_event = threading.Event()
        self.pending_requests = set()
        self.pending_requests_lock = threading.Lock()
        
        # Coalescing statistics
        self.coalesced_requests = 0
        self.direct_batches_processed = 0
        self.total_coalesced_queries = 0
        self.stats_lock = threading.Lock()
        
        # Sample dataset (will be replaced by actual dataset if provided)
        self.sample_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the process of photosynthesis.",
            "What are the main causes of climate change?",
            "How do neural networks function?",
            "Explain the theory of relativity.",
            "What is the significance of DNA?"
        ]
        self.sample_token_ids = []
        if self.dataset_path:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path, total_sample_count=13368)
            logger.info("Dataset loaded: %d samples", len(self.data_object.input_ids))
            logger.info("Dataset statistics - Max Input Tokens: %d, Min Input Tokens: %d, Total Samples: %d", 
                            max(self.data_object.input_lens), 
                            min(self.data_object.input_lens),
                            len(self.data_object.input_lens))
        else:
            logger.info("No dataset path provided, will use sample prompts")
            self.data_object = None 
        logger.info(f"MLPerf SUT initialized for model: {model_path}")
        
        # Don't start workers in init - they'll be started when the engine is ready
        self._workers_started = False
    
    async def initialize_engine(self):
        """Prepare for AsyncLLMEngine initialization (will be done in async processor thread)."""
        logger.info("Preparing for AsyncLLMEngine initialization...")
        
        # Just prepare sample prompts for now - engine will be initialized in async processor
        # Create a temporary tokenizer to tokenize sample prompts
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize sample prompts
        self.sample_token_ids = []
        for prompt in self.sample_prompts:
            token_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            self.sample_token_ids.append(token_ids)
            logger.info(f"Prompt: '{prompt[:30]}...' -> {len(token_ids)} tokens")
        
        logger.info("Sample prompts tokenized. Engine will be initialized in async processor thread.")
    
    def start_workers(self):
        """Start worker threads and async processor for MLPerf queries."""
        logger.info(f"Starting {self.num_workers} worker threads and async processor for MLPerf SUT")
        self.is_running = True
        
        # Start the single async processor thread
        self.async_processor_thread = threading.Thread(target=self._async_processor_loop, name="AsyncProcessor")
        self.async_processor_thread.daemon = True
        self.async_processor_thread.start()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"MLPerf-Worker-{i}")
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
    
    def stop_workers(self):
        """Stop worker threads and async processor."""
        logger.info("Stopping MLPerf SUT worker threads and async processor...")
        self.is_running = False
        
        # Send stop signals to workers
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        
        # Send stop signal to async processor
        self.response_queue.put(("STOP", None))
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join()
        
        # Wait for async processor to finish
        if self.async_processor_thread:
            self.async_processor_thread.join()
        
        # Gracefully shutdown the async engine if it exists
        if hasattr(self, 'llm_engine') and self.llm_engine and hasattr(self.llm_engine, 'engine'):
            logger.info("Shutting down AsyncLLMEngine...")
            try:
                # Get the existing event loop from the async processor thread
                engine = self.llm_engine.engine
                
                # Cancel any pending tasks in the engine
                if hasattr(engine, '_engine_loop') and engine._engine_loop:
                    logger.info("Attempting to cancel engine tasks...")
                    try:
                        # Get all tasks in the engine's loop
                        all_tasks = asyncio.all_tasks(engine._engine_loop)
                        logger.info(f"Found {len(all_tasks)} tasks to cancel")
                        
                        for task in all_tasks:
                            if not task.done():
                                task.cancel()
                                logger.debug(f"Cancelled task: {task}")
                    except Exception as e:
                        logger.warning(f"Error cancelling tasks: {e}")
                
                # Give a moment for tasks to cancel
                import time
                time.sleep(0.5)
                logger.info("AsyncLLMEngine shutdown completed")
                
            except Exception as e:
                logger.warning(f"Error during AsyncLLMEngine shutdown: {e}")
        
        logger.info("All threads stopped")
    
    def _async_processor_loop(self):
        """Single async processor thread that handles all AsyncLLMEngine operations."""
        logger.info("Async processor thread started")
        
        # Create dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_processor_main())
        except Exception as e:
            logger.error(f"Error in async processor: {e}")
        finally:
            loop.close()
            logger.info("Async processor thread stopped")
    
    async def _async_processor_main(self):
        """Main async processor loop."""
        # Initialize the engine in this thread/event loop
        logger.info("Initializing AsyncLLMEngine in async processor thread...")
        try:
            # Create AsyncLLMEngine instance
            self.llm_engine = SimpleAsyncLLMEngine(
                model_path=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len
            )
            
            # Initialize the engine
            await self.llm_engine.initialize()
            
            # Signal that engine is ready
            self.engine_ready_event.set()
            
            logger.info("AsyncLLMEngine initialized successfully in async processor thread")
            
        except Exception as e:
            logger.error(f"Failed to initialize AsyncLLMEngine in async processor: {e}")
            return
        
        # Main processing loop
        while self.is_running:
            try:
                # Get processing request from queue with timeout
                def get_with_timeout():
                    try:
                        return self.response_queue.get(timeout=1)
                    except queue.Empty:
                        return None
                
                item = await asyncio.get_event_loop().run_in_executor(None, get_with_timeout)
                
                if item is None:
                    continue
                    
                command, data = item
                
                if command == "STOP":
                    logger.info("Async processor received stop signal")
                    break
                elif command == "PROCESS":
                    # Legacy single query processing (kept for compatibility)
                    query_sample, token_ids = data
                    logger.info(f"Async processor handling single query {query_sample.id}")
                    await self._process_single_async(query_sample, token_ids)
                elif command == "PROCESS_BATCH":
                    # New batch processing
                    batch = data
                    batch_size = len(batch)
                    logger.info(f"🔄 Async processor handling batch of {batch_size} queries")
                    
                    # Log query IDs for debugging
                    query_ids = [str(q[0].id) for q in batch[:3]]  # First 3 IDs
                    if batch_size > 3:
                        query_ids.append(f"...+{batch_size-3} more")
                    logger.debug(f"Batch query IDs: {', '.join(query_ids)}")
                    
                    await self._process_batch_async(batch)
                    
            except Exception as e:
                logger.error(f"Error in async processor main loop: {e}")
                continue
    
    def _worker_loop(self):
        """Main worker loop that collects queries into batches with timeout."""
        logger.info(f"Worker thread {threading.current_thread().name} started with batch_size={self.batch_size}, timeout={self.batch_timeout_ms}ms")
        
        while self.is_running:
            try:
                batch = []
                batch_start_time = time.time()
                timeout_seconds = self.batch_timeout_ms / 1000.0
                
                # Collect queries until batch_size or timeout
                while len(batch) < self.batch_size and self.is_running:
                    remaining_time = timeout_seconds - (time.time() - batch_start_time)
                    
                    if remaining_time <= 0 and batch:
                        # Timeout reached and we have at least one query
                        logger.debug(f"Batch timeout reached with {len(batch)} queries")
                        break
                    
                    try:
                        # Use remaining timeout or minimum 0.1s
                        wait_time = max(0.1, remaining_time) if batch else 1.0
                        query_sample = self.query_queue.get(timeout=wait_time)
                        
                        if query_sample is None:  # Stop signal
                            if batch:
                                # Process remaining batch before stopping
                                logger.info(f"Worker {threading.current_thread().name} processing final batch of {len(batch)} queries before stop")
                                break
                            else:
                                logger.info(f"Worker {threading.current_thread().name} received stop signal")
                                return
                        
                        # Add query to batch
                        if self.data_object:
                            # Use real dataset with bounds checking
                            if query_sample.index < len(self.data_object.input_ids):
                                token_ids = self.data_object.input_ids[query_sample.index]
                            else:
                                logger.warning(f"Query index {query_sample.index} out of bounds for dataset (size: {len(self.data_object.input_ids)}), using modulo")
                                token_ids = self.data_object.input_ids[query_sample.index % len(self.data_object.input_ids)]
                        else:
                            # Use sample prompts
                            sample_idx = query_sample.index % len(self.sample_token_ids)
                            token_ids = self.sample_token_ids[sample_idx]
                        
                        batch.append((query_sample, token_ids))
                        
                        logger.debug(f"Added query {query_sample.id} to batch (size: {len(batch)}/{self.batch_size})")
                        
                    except queue.Empty:
                        if batch:
                            # Timeout reached with queries in batch
                            logger.debug(f"Queue timeout with {len(batch)} queries in batch")
                            break
                        else:
                            # No queries available, continue outer loop
                            continue
                
                # Process the collected batch
                if batch:
                    batch_time = (time.time() - batch_start_time) * 1000
                    logger.info(f"Worker sending batch of {len(batch)} queries (collected in {batch_time:.1f}ms)")
                    self.response_queue.put(("PROCESS_BATCH", batch))
                        
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                # Send error responses for any queries in current batch
                if 'batch' in locals():
                    for query_sample, _ in batch:
                        try:
                            response = lg.QuerySampleResponse(query_sample.id, 0, 0, 0)
                            lg.QuerySamplesComplete([response])
                        except:
                            pass
                
        logger.info(f"Worker thread {threading.current_thread().name} stopped")
    
    async def _process_single_async(self, query_sample, token_ids):
        """Process a single query asynchronously."""
        request_id = f"mlperf_{query_sample.id}_{self.request_id}"
        self.request_id += 1
        
        # Track pending request
        with self.pending_requests_lock:
            self.pending_requests.add(request_id)
        
        try:
            logger.info(f"Starting async processing for query {query_sample.id}")
            
            # Convert to vLLM format
            vllm_prompt = TokensPrompt(prompt_token_ids=token_ids)
            
            logger.info(f"Query {query_sample.id}: Calling engine.generate with request_id {request_id}")
            
            results_generator = self.llm_engine.engine.generate(
                prompt=vllm_prompt,
                sampling_params=self.llm_engine.sampling_params,
                request_id=request_id
            )
            
            logger.info(f"Query {query_sample.id}: Got results generator, starting iteration")
            
            # Stream and capture the generated text
            generated_token_ids = []
            first_token_reported = False
            start_time = time.time()
            iteration_count = 0
            
            async for request_output in results_generator:
                iteration_count += 1
                logger.debug(f"Query {query_sample.id}: Iteration {iteration_count}")
                
                if request_output.outputs:
                    current_token_ids = request_output.outputs[0].token_ids
                    
                    # Report first token if not already reported
                    if not first_token_reported and current_token_ids:
                        first_token_reported = True
                        
                        logger.info(f"Query {query_sample.id}: Reporting first token")
                        
                        # Create response data for first token
                        response_data = array.array("B", np.array(current_token_ids, np.int32).tobytes())
                        bi = response_data.buffer_info()
                        response = [lg.QuerySampleResponse(query_sample.id, bi[0], bi[1])]
                        lg.FirstTokenComplete(response)
                        
                        first_token_latency = time.time() - start_time
                        logger.info(f"Query {query_sample.id} first token reported in {first_token_latency:.3f}s")
                    
                    generated_token_ids = current_token_ids
            
            logger.info(f"Query {query_sample.id}: Finished iteration after {iteration_count} steps")
            
            # Report final completion
            if generated_token_ids:
                total_time = time.time() - start_time
                
                if self.test_mode == "accuracy":
                    # Accuracy mode: send full response data
                    response_data = array.array("B", np.array(generated_token_ids, np.int32).tobytes())
                    bi = response_data.buffer_info()
                    response = [lg.QuerySampleResponse(query_sample.id, bi[0], bi[1], len(generated_token_ids))]
                    logger.info(f"Query {query_sample.id} completed (accuracy mode)! {len(generated_token_ids)} tokens in {total_time:.2f}s")
                else:
                    # Performance mode: only report token count
                    response = [lg.QuerySampleResponse(query_sample.id, 0, 0, len(generated_token_ids))]
                    logger.info(f"Query {query_sample.id} completed (performance mode)! {len(generated_token_ids)} tokens in {total_time:.2f}s")
                
                lg.QuerySamplesComplete(response)
            else:
                # Empty response
                response = [lg.QuerySampleResponse(query_sample.id, 0, 0, 0)]
                lg.QuerySamplesComplete(response)
                logger.warning(f"Query {query_sample.id} generated empty response")
                
        except Exception as e:
            logger.error(f"Error processing query {query_sample.id}: {e}")
            # Send error response
            response = [lg.QuerySampleResponse(query_sample.id, 0, 0, 0)]
            lg.QuerySamplesComplete(response)
        finally:
            # Remove from pending requests
            with self.pending_requests_lock:
                self.pending_requests.discard(request_id)
            logger.debug(f"Request {request_id} completed and removed from pending set")
    
    async def _process_batch_async(self, batch):
        """Process a batch of queries concurrently."""
        logger.info(f"Starting concurrent processing of {len(batch)} queries")
        
        # Create concurrent tasks for all queries in the batch
        tasks = []
        for query_sample, token_ids in batch:
            task = asyncio.create_task(
                self._process_single_async(query_sample, token_ids)
            )
            tasks.append(task)
        
        # Process all queries concurrently
        try:
            await asyncio.gather(*tasks)
            logger.info(f"Completed concurrent processing of {len(batch)} queries")
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Error handling is done in individual _process_single_async calls

    
    # MLPerf LoadGen Interface Methods
    def issue_query(self, query_samples):
        """MLPerf LoadGen callback: Issue queries to the SUT."""
        num_queries = len(query_samples)
        logger.info(f"MLPerf LoadGen issued {num_queries} queries")
        
        if num_queries > 1:
            # Coalesced request - log and potentially batch directly
            with self.stats_lock:
                self.coalesced_requests += 1
                self.total_coalesced_queries += num_queries
            
            logger.info(f"🔄 Coalesced request detected: {num_queries} queries in single call")
            
            # If coalesced batch is smaller than or equal to our batch size, process directly
            if num_queries <= self.batch_size:
                logger.info(f"✅ Processing coalesced batch of {num_queries} queries directly (bypassing worker queue)")
                
                # Prepare the batch directly
                batch = []
                for query_sample in query_samples:
                    if self.data_object:
                        # Use real dataset with bounds checking
                        if query_sample.index < len(self.data_object.input_ids):
                            token_ids = self.data_object.input_ids[query_sample.index]
                        else:
                            logger.warning(f"Query index {query_sample.index} out of bounds for dataset (size: {len(self.data_object.input_ids)}), using modulo")
                            token_ids = self.data_object.input_ids[query_sample.index % len(self.data_object.input_ids)]
                    else:
                        # Use sample prompts
                        sample_idx = query_sample.index % len(self.sample_token_ids)
                        token_ids = self.sample_token_ids[sample_idx]
                    
                    batch.append((query_sample, token_ids))
                    logger.debug(f"Added coalesced query {query_sample.id} with index {query_sample.index}")
                
                # Send directly to async processor as a batch
                self.response_queue.put(("PROCESS_BATCH", batch))
                
                with self.stats_lock:
                    self.direct_batches_processed += 1
                    
                logger.info(f"📤 Sent coalesced batch of {num_queries} queries directly to async processor")
                return
            elif num_queries > self.batch_size:
                # Large coalesced batch - split into optimal chunks
                logger.info(f"📦 Large coalesced batch ({num_queries} queries) - splitting into chunks of {self.batch_size}")
                
                for i in range(0, num_queries, self.batch_size):
                    chunk_end = min(i + self.batch_size, num_queries)
                    chunk = query_samples[i:chunk_end]
                    chunk_size = len(chunk)
                    
                    logger.info(f"Processing chunk {i//self.batch_size + 1}: {chunk_size} queries (indices {i}-{chunk_end-1})")
                    
                    # Prepare the chunk as a batch
                    batch = []
                    for query_sample in chunk:
                        if self.data_object:
                            # Use real dataset with bounds checking
                            if query_sample.index < len(self.data_object.input_ids):
                                token_ids = self.data_object.input_ids[query_sample.index]
                            else:
                                logger.warning(f"Query index {query_sample.index} out of bounds for dataset (size: {len(self.data_object.input_ids)}), using modulo")
                                token_ids = self.data_object.input_ids[query_sample.index % len(self.data_object.input_ids)]
                        else:
                            # Use sample prompts
                            sample_idx = query_sample.index % len(self.sample_token_ids)
                            token_ids = self.sample_token_ids[sample_idx]
                        
                        batch.append((query_sample, token_ids))
                    
                    # Send chunk directly to async processor
                    self.response_queue.put(("PROCESS_BATCH", batch))
                    
                    with self.stats_lock:
                        self.direct_batches_processed += 1
                
                logger.info(f"📤 Sent {num_queries} coalesced queries as {(num_queries + self.batch_size - 1) // self.batch_size} chunks to async processor")
                return
            else:
                logger.info(f"⚠️ Unusual coalesced batch size {num_queries}, using worker queue")
        
        # Regular processing: queue individual queries for worker threads to batch
        for query_sample in query_samples:
            logger.debug(f"Queuing query {query_sample.id} with index {query_sample.index}")
            self.query_queue.put(query_sample)
        
        logger.debug(f"All {num_queries} queries queued. Queue size: {self.query_queue.qsize()}")
    
    def print_coalescing_stats(self):
        """Print statistics about coalescing effectiveness."""
        with self.stats_lock:
            if self.coalesced_requests > 0:
                avg_coalesced_size = self.total_coalesced_queries / self.coalesced_requests
                logger.info("📊 === Coalescing Statistics ===")
                logger.info(f"📦 Coalesced requests: {self.coalesced_requests}")
                logger.info(f"📈 Total coalesced queries: {self.total_coalesced_queries}")
                logger.info(f"📊 Average coalesced batch size: {avg_coalesced_size:.1f}")
                logger.info(f"⚡ Direct batches processed: {self.direct_batches_processed}")
                logger.info(f"🎯 Coalescing efficiency: {(self.direct_batches_processed/self.coalesced_requests*100):.1f}% batches processed directly")
            else:
                logger.info("📊 No coalesced requests received")
    
    def flush_queries(self):
        """MLPerf LoadGen callback: Flush any pending queries."""
        logger.info("MLPerf LoadGen flush queries called - waiting for pending requests to complete")
        
        # Wait for all pending requests to complete
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while True:
            with self.pending_requests_lock:
                pending_count = len(self.pending_requests)
                
            if pending_count == 0:
                logger.info("All requests completed successfully")
                break
                
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.warning(f"Timeout waiting for {pending_count} pending requests after {elapsed:.1f}s")
                with self.pending_requests_lock:
                    logger.warning(f"Remaining pending requests: {list(self.pending_requests)}")
                break
                
            logger.info(f"Waiting for {pending_count} pending requests... ({elapsed:.1f}s elapsed)")
            time.sleep(0.5)
        
        # Print coalescing statistics
        self.print_coalescing_stats()
        
        logger.info("Flush queries completed")


def run_mlperf_test(sut, num_samples=100, target_qps=10, max_duration_ms=60000, args=None):
    """Run MLPerf LoadGen test with the given SUT."""
    logger.info("Setting up MLPerf LoadGen test...")
    
    # Determine actual sample count based on dataset availability
    if sut.data_object:
        actual_num_samples = min(num_samples, len(sut.data_object.input_ids))
        logger.info(f"Using dataset with {actual_num_samples} samples (requested: {num_samples}, available: {len(sut.data_object.input_ids)})")
    else:
        actual_num_samples = num_samples
        logger.info(f"Using sample prompts with {actual_num_samples} samples")
    
    # Create test settings
    test_settings = lg.TestSettings()
    test_settings.scenario = lg.TestScenario.Server
    
    # Set mode based on SUT test_mode
    if sut.test_mode == "accuracy":
        test_settings.mode = lg.TestMode.AccuracyOnly
        logger.info("Running in Accuracy mode")
    else:
        test_settings.mode = lg.TestMode.PerformanceOnly
        logger.info("Running in Performance mode")
    
    # Server scenario settings
    test_settings.server_target_qps = target_qps
    test_settings.server_target_latency_ns = 100_000_000  # 100ms
    test_settings.server_target_latency_percentile = 0.99
    test_settings.max_duration_ms = max_duration_ms
    test_settings.min_duration_ms = 10000  # 10 seconds minimum
    
    # Server coalesce setting
    if args and hasattr(args, 'enable_coalesce') and args.enable_coalesce:
        test_settings.server_coalesce_queries = True
        logger.info("Server coalesce enabled")
    
    # Max async queries setting
    if args and hasattr(args, 'max_async_queries') and args.max_async_queries > 0:
        test_settings.max_async_queries = args.max_async_queries
        logger.info(f"Max async queries set to: {args.max_async_queries}")
    
    # Log settings
    log_settings = lg.LogSettings()
    log_dir = args.log_dir if args and hasattr(args, 'log_dir') else "."
    log_settings.log_output.outdir = log_dir
    log_settings.log_output.prefix = "mlperf_log_async_llm"
    log_settings.log_output.suffix = ""
    log_settings.log_output.prefix_with_datetime = True
    log_settings.log_output.copy_detail_to_stdout = False  # Reduce stdout noise
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.log_mode = lg.LoggingMode.AsyncPoll
    log_settings.log_mode_async_poll_interval_ms = 1000
    log_settings.enable_trace = False  # Reduce overhead
    
    logger.info(f"MLPerf logs will be written to: {log_dir}")
    
    # Create query sample library (QSL)
    class SimpleQSL:
        def __init__(self, total_sample_count, performance_sample_count=None):
            self.total_sample_count = total_sample_count
            # Performance sample count can be smaller for faster loading
            self.performance_sample_count = performance_sample_count or min(total_sample_count, 1024)
            logger.info(f"QSL: Total samples={self.total_sample_count}, Performance samples={self.performance_sample_count}")
            
        def load_samples_to_ram(self, sample_indices):
            # No-op for this demo - data is already in memory
            logger.debug(f"QSL: Loading {len(sample_indices)} samples to RAM")
            pass
            
        def unload_samples_from_ram(self, sample_indices):
            # No-op for this demo
            logger.debug(f"QSL: Unloading {len(sample_indices)} samples from RAM")
            pass
            
        def get_sample_count(self):
            return self.total_sample_count
            
        def get_performance_sample_count(self):
            return self.performance_sample_count
            
        def get_sample(self, sample_index):
            # Return empty sample - we'll use the sample index to select from our predefined prompts
            return ""
    
    # Determine performance sample count
    performance_samples = None
    if args and hasattr(args, 'performance_samples') and args.performance_samples:
        performance_samples = args.performance_samples
    qsl = SimpleQSL(actual_num_samples, performance_samples)
    
    # Register QSL and SUT with LoadGen
    logger.info("Constructing QSL...")
    QSLToTest = lg.ConstructQSL(
        qsl.get_sample_count(),
        qsl.get_performance_sample_count(),  # Performance sample count (can be smaller)
        qsl.load_samples_to_ram,
        qsl.unload_samples_from_ram
    )
    logger.info(f"QSL constructed with ID: {QSLToTest}")
    
    logger.info("Constructing SUT...")
    SUTToTest = lg.ConstructSUT(
        sut.issue_query,
        sut.flush_queries
    )
    logger.info(f"SUT constructed with ID: {SUTToTest}")
    
    logger.info(f"Starting MLPerf test - Server scenario")
    logger.info(f"Target QPS: {target_qps}, Max duration: {max_duration_ms}ms")
    logger.info(f"Number of worker threads: {sut.num_workers}")
    logger.info(f"Batch size: {sut.batch_size}")
    logger.info(f"Batch timeout: {sut.batch_timeout_ms}ms")
    logger.info(f"QSL Total samples: {qsl.get_sample_count()}")
    logger.info(f"QSL Performance samples: {qsl.get_performance_sample_count()}")
    
    try:
        # Run the test
        logger.info("Calling StartTestWithLogSettings...")
        lg.StartTestWithLogSettings(SUTToTest, QSLToTest, test_settings, log_settings)
        
        logger.info("MLPerf test completed!")
        
    finally:
        # Clean up
        logger.info("Cleaning up...")
        sut.stop_workers()
        lg.DestroySUT(SUTToTest)
        lg.DestroyQSL(QSLToTest)
        logger.info("Cleanup completed")


def main():
    """Main function to demonstrate the AsyncLLMEngine usage."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AsyncLLMEngine Demo with MLPerf LoadGen Support")
    parser.add_argument("--model", "-m", type=str, default="gpt2",
                       help="Model path or HuggingFace model name (default: gpt2)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                       help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--max-model-len", "-ml", type=int, default=1024,
                       help="Maximum sequence length (default: 1024)")
    parser.add_argument("--use-text", action="store_true",
                       help="Use text prompts instead of token IDs (demo mode only)")
    
    # MLPerf specific arguments
    parser.add_argument("--mlperf", action="store_true",
                       help="Run MLPerf LoadGen test instead of demo")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for MLPerf processing (default: 4)")
    parser.add_argument("--batch-timeout", type=float, default=10.0,
                       help="Batch timeout in milliseconds (default: 10.0)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of worker threads for MLPerf (default: 1)")
    parser.add_argument("--target-qps", type=float, default=10.0,
                       help="Target QPS for MLPerf Server scenario (default: 10.0)")
    parser.add_argument("--max-duration", type=int, default=60000,
                       help="Max test duration in milliseconds (default: 60000)")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples for MLPerf test (default: 100)")
    parser.add_argument("--performance-samples", type=int, default=None,
                       help="Number of performance samples to load (default: min(num_samples, 1024))")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to dataset file (default: use sample prompts)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    # MLPerf specific settings
    parser.add_argument("--enable-coalesce", action="store_true",
                       help="Enable server coalesce in MLPerf LoadGen")
    parser.add_argument("--log-dir", type=str, default=".",
                       help="Directory for MLPerf log outputs (default: current directory)")
    parser.add_argument("--accuracy-mode", action="store_true",
                       help="Run in accuracy mode (send full response data)")
    parser.add_argument("--max-async-queries", type=int, default=0,
                       help="Maximum async queries for MLPerf LoadGen (default: 0 = auto)")
    parser.add_argument("--cuda-devices", type=str, default=None,
                       help="CUDA_VISIBLE_DEVICES setting (e.g., '0,1,2,3')")
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if specified
    import os
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda_devices}")
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Check for environment variable override, otherwise use command line args
    model_path = os.getenv("MODEL_PATH", args.model)
    
    logger.info(f"Using model: {model_path}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"Max model length: {args.max_model_len}")
    
    if args.mlperf:
        logger.info("=== MLPerf LoadGen Mode ===")
        logger.info(f"Test mode: {args.accuracy_mode and 'Accuracy' or 'Performance'}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Batch timeout: {args.batch_timeout}ms")
        logger.info(f"Workers: {args.num_workers}")
        logger.info(f"Target QPS: {args.target_qps}")
        logger.info(f"Max duration: {args.max_duration}ms")
        logger.info(f"Num samples: {args.num_samples}")
        logger.info(f"Performance samples: {args.performance_samples if args.performance_samples else 'Auto'}")
        logger.info(f"Dataset path: {args.dataset_path if args.dataset_path else 'Using sample prompts'}")
        logger.info(f"Log directory: {args.log_dir}")
        logger.info(f"Server coalesce: {args.enable_coalesce}")
        logger.info(f"Max async queries: {args.max_async_queries if args.max_async_queries > 0 else 'Auto'}")
        if args.cuda_devices:
            logger.info(f"CUDA devices: {args.cuda_devices}")
        
        # Create MLPerf SUT
        sut = MLPerfAsyncSUT(
            model_path=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            batch_timeout_ms=args.batch_timeout,
            dataset_path=args.dataset_path,
            test_mode="accuracy" if args.accuracy_mode else "performance"
        )
        
        try:
            # Prepare for engine initialization (tokenize sample prompts)
            asyncio.run(sut.initialize_engine())
            
            # Start workers - engine will be initialized in async processor thread
            sut.start_workers()
            
            # Wait for async processor to initialize engine
            logger.info("Waiting for async processor to initialize engine...")
            if not sut.engine_ready_event.wait(timeout=300):  # Wait up to 300 seconds
                raise RuntimeError("AsyncLLMEngine failed to initialize within 300seconds")
            logger.info("AsyncLLMEngine is ready!")
            
            # Run MLPerf test
            run_mlperf_test(
                sut=sut,
                num_samples=args.num_samples,
                target_qps=args.target_qps,
                max_duration_ms=args.max_duration,
                args=args
            )
            
        except Exception as e:
            logger.error(f"Error running MLPerf test: {e}")
            raise
    
    else:
        logger.info("=== Demo Mode ===")
        logger.info(f"Input format: {'Text' if args.use_text else 'Token IDs'}")
        
        # Create and initialize the engine
        llm_engine = SimpleAsyncLLMEngine(
            model_path=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len
        )
        
        try:
            # Initialize the engine
            asyncio.run(llm_engine.initialize())
            
            # Run sample generation
            asyncio.run(llm_engine.run_sample_generation(use_token_ids=not args.use_text))
            
        except Exception as e:
            logger.error(f"Error running AsyncLLMEngine: {e}")
            raise


if __name__ == "__main__":
    # Run the main function
    main()
