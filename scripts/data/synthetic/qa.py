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
# limitations under the License

"""
Create a dataset jsonl file for QA task.

python qa.py \
    --save_dir=./ \
    --save_name=niah_single \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --max_seq_length=4096 \
    --tokens_to_generate=128 \
    --num_samples=10 \
    --template="Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query} Answer:"
"""
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 
from tokenizer import select_tokenizer


parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--pre_samples", type=int, default=0, help='number of samples are already generated')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, required=True, help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

# Complexity Configurations
parser.add_argument("--dataset", type=str, required=True, help='dataset file')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
                        
    return total_qas, total_docs

# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
        
    return total_qas, total_docs


DOCUMENT_PROMPT = "Document {i}:\n{document}"
if args.dataset == 'squad':
    QAS, DOCS = read_squad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/squad.json"))
elif args.dataset == 'hotpotqa':
    QAS, DOCS = read_hotpotqa(os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/hotpotqa.json"))
else:
    raise NotImplementedError(f'{args.dataset} is not implemented.')


def generate_input_output(index, num_docs):
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
    
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS
        
    random.Random(args.random_seed).shuffle(all_docs)
    
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    input_text = args.template.format(
        context=context, 
        query=curr_q
    )
    return input_text, curr_a


def generate_samples(num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 10): 
    
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    
    # Find the perfect num_docs
    num_docs = incremental
    
    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = generate_input_output(0, num_docs)
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.text_to_tokens(input_text + f' {answer}'))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break
            
        num_docs += incremental
        if num_docs > len(DOCS):
            num_docs = len(DOCS)
            break
    print('Number of documents:', num_docs)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while(True):
            try:
                input_text, answer = generate_input_output(index + args.pre_samples, used_docs)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_docs > incremental:
                    used_docs -= incremental
        
        if args.remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
        
        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples, 
        max_seq_length=args.max_seq_length, 
        save_dir=args.save_dir
    )
    
    write_manifest(save_file, write_jsons)

if __name__=="__main__":
    main()
