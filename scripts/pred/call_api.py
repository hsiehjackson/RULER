# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
    args.threads = 1


def get_llm(tokens_to_generate):
    if args.server_type == 'trtllm':
        from client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'sglang':
        from client_wrappers import SGLClient
        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
        
    elif args.server_type == 'hf':
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    
    elif args.server_type == 'mamba':
        from model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
        
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def main():
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    # Load api
    llm = get_llm(config['tokens_to_generate'])

    def get_output(outputs_parallel, idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        while True:
            print("In processing loop")
            try:
                pred_list = llm.process_batch(prompts=input_list)
                break
            except Exception as e:
                traceback.print_exc()

        print("exited processing loop")
        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }
        return outputs_parallel

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    print(f"Total data: {len(data)}")
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    print("exited data loop")
    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        print(f"Total batches: {len(batched_data)}")
        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                )
            outputs_parallel = get_output(outputs_parallel, **kwargs)
            # print("begin threading")
            # thread = threading.Threa
            #     target=get_output,
            #     kwargs=dict(
            #         idx_list=idx_list,
            #         index_list=[data_point['index'] for data_point in batch],
            #         input_list=[data_point['input'] for data_point in batch],
            #         outputs_list=[data_point['outputs'] for data_point in batch],
            #         others_list=[data_point.get('others', {}) for data_point in batch],
            #         truncation_list=[data_point.get('truncation', -1) for data_point in batch],
            #         length_list=[data_point.get('length', -1) for data_point in batch],
            #     ),
            # )
            # thread.start()
            # threads.append(thread)
            # print("middle of threading stuff, before threads join")

            # is_last_batch = (batch_idx == len(batched_data) - 1)
            # print(f"Is last batch: {is_last_batch}")
            # print(f"Length of threads: {len(threads)}")
            # if (len(threads) == args.threads) or is_last_batch:
            #     print("In join block")
            #     for thread in threads:
            #         print("before join")
            #         thread.join()
            #         print("after join")
            #     threads = []
            #     print("After threads join")

            #     # dump the results in current processing window on disk
            #     for idx in range(start_idx, end_idx + 1):
            #         if len(outputs_parallel[idx]) > 0:
            #             fout.write(json.dumps(outputs_parallel[idx]) + '\n')

            #     start_idx = end_idx + 1
            
            for idx in range(start_idx, end_idx + 1):
                if len(outputs_parallel[idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[idx]) + '\n')

            start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
