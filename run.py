import os
import subprocess
import sys
import argparse
from datetime import datetime

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run RULER benchmark.')
    parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-chat-hf", type=str, help='Name of the model')
    parser.add_argument('--num_tokens', default=4096, type=int, help='Number of tokens')
    parser.add_argument('--task_name', default='niah_single_1', type=str, help='Name of the task')
    args = parser.parse_args()

    # Root Directories
    ROOT_DIR = "benchmark_root"  # the path that stores generated task samples and model predictions.
    BATCH_SIZE = 1  # increase to improve GPU utilization

    # Model and Tokenizer
    MODEL_NAME = args.model_name
    TEMPERATURE = "0.0"  # greedy
    TOP_P = "1.0"
    TOP_K = "32"
    MODEL_PATH = args.model_name
    # MODEL_TEMPLATE_TYPE = "base"
    MODEL_TEMPLATE_TYPE = "meta-chat"
    MODEL_FRAMEWORK = "hf"
    TOKENIZER_PATH = MODEL_PATH
    TOKENIZER_TYPE = "hf"

    # Benchmark and Tasks
    NUM_SAMPLES = 4
    BENCHMARK = "synthetic"

    # Start client (prepare data / call model API / obtain final metrics)
    RESULTS_DIR = f"{ROOT_DIR}/{MODEL_NAME}/{BENCHMARK}/{args.num_tokens}"
    DATA_DIR = f"{RESULTS_DIR}/data"
    PRED_DIR = f"{RESULTS_DIR}/pred"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    subprocess.run([
        "python", "scripts/data/prepare.py",
        "--save_dir", DATA_DIR,
        "--benchmark", BENCHMARK,
        "--task", args.task_name,
        "--tokenizer_path", TOKENIZER_PATH,
        "--tokenizer_type", TOKENIZER_TYPE,
        "--max_seq_length", str(args.num_tokens),
        "--model_template_type", MODEL_TEMPLATE_TYPE,
        "--num_samples", str(NUM_SAMPLES)
    ])

    subprocess.run([
        "python", "scripts/pred/call_api.py",
        "--data_dir", DATA_DIR,
        "--save_dir", PRED_DIR,
        "--benchmark", BENCHMARK,
        "--task", args.task_name,
        "--server_type", MODEL_FRAMEWORK,
        "--model_name_or_path", MODEL_PATH,
        "--temperature", TEMPERATURE,
        "--top_k", TOP_K,
        "--top_p", TOP_P,
        "--batch_size", str(BATCH_SIZE),
    ])

    subprocess.run([
        "python", "scripts/eval/evaluate.py",
        "--data_dir", PRED_DIR,
        "--benchmark", BENCHMARK
    ])

if __name__ == "__main__":
    main()