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
Create a dataset jsonl file for variable tracking.

python variable_tracking.py   \
    --save_dir=./ \
    --save_name=vt \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --tokens_to_generate 30 \
    --num_samples 10 \
    --random_seed 42  \
    --num_chains 1  --num_hops 4 \
    --template "[INST] Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above. [/INST] Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: "
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import string
from constants import TASKS
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 
from tokenizer import select_tokenizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, default=120, help='number of tokens to generate')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default='', help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

parser.add_argument("--num_chains", type=int, default=1, help='number of inserted variable chains')
parser.add_argument("--num_hops", type=int, default=4, help='number of hops in each chain')
parser.add_argument("--add_fewshot", action="store_true", default=False)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

def generate_chains(num_chains, num_hops, is_icl=False):
    
    vars_all = []
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [''.join(random.choices(string.ascii_uppercase, k=k)).upper() for _ in range((num_hops+1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops+1):
        vars_all.append(''.join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops+1):
        this_vars = vars_all[i:i+num_hops+1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret
    
def generate_input_output(num_noises, num_chains, num_hops, is_icl=False):

    vars, chains = generate_chains(num_chains, num_hops, is_icl=is_icl)

    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"

    # Create a list of the repeated noise
    sentences = [noise] * num_noises
    if len(sentences) <= len(chains[0]):
        sentences = [n + '.' if len(n.strip()) > 0 else n for n in [x for noise in sentences for x in noise.split('.')] ]
        try:
            assert len(sentences) > len(chains[0]), "Noises too short, unable to generate data"
        except:
            print("reduces chain length for not enough noises")
            chains = [chain[:len(sentences)-1] for chain in chains]
    # sample random positions to insert variable assignment
    for chain_i in chains:
        # sample random positions (sorted) to insert variable assignment
        positions = list(sorted(random.sample(range(len(sentences)), len(chain_i))))
        for insert_pi, j in zip(positions, range(len(chain_i))):
            sentences.insert(insert_pi+j, chain_i[j])

    # Insert the passkey sentence at the random position
    context = " ".join(sentences)
    context = context.replace(". \n", ".\n")

    template = args.template
    if is_icl:
        # remove model template
        cutoff = template.index(TASKS['variable_tracking']['template'][:20])
        cutoff_ans = template.index(TASKS['variable_tracking']['answer_prefix'][:10])
        template = ' '.join(template[cutoff:cutoff_ans].split()[:-1]) + template[cutoff_ans:]

    value = chains[0][0].split("=")[-1].strip()
    input_text = template.format(
        context=context,
        query=value,
        num_v=num_hops+1
    )

    return input_text, vars[0]


def sys_vartrack_w_noise_random(num_samples: int, max_seq_length: int, incremental: int = 10, 
                                num_chains: int = 1, num_hops: int = 4,
                                add_fewshot: bool = True,
                                icl_example: str = None):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    
    # Find the perfect num_noises
    num_noises = incremental
        
    total_tokens = 0  # Track the total tokens generated for this example
    example_tokens = 0
    if add_fewshot and (icl_example is not None):
        icl_example_out = ' '.join(icl_example['outputs'])
        icl_example = icl_example['input'] + " " + icl_example_out + '\n\n'
        example_tokens = len(TOKENIZER.text_to_tokens(icl_example)) 
        
    while total_tokens + tokens_to_generate + example_tokens < max_seq_length :
        input_text, answer = generate_input_output(num_noises, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None))
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.text_to_tokens(input_text + f' {answer}'))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate + example_tokens} | Noises: {num_noises}')
        if total_tokens + tokens_to_generate + example_tokens > max_seq_length:
            num_noises -= incremental
            break
        num_noises += incremental
    print('Num noises:', num_noises)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_noises = num_noises
        while(True):
            try:
                input_text, answer = generate_input_output(num_noises, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None))
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate + example_tokens
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_noises > incremental:
                    used_noises -= incremental

        if add_fewshot and (icl_example is not None):
            # insert icl_example between model template and input
            cutoff = input_text.index(TASKS['variable_tracking']['template'][:20])
            input_text = input_text[:cutoff] + ' ' + icl_example + '\n\n' + input_text[cutoff:]
        if args.remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
        
        formatted_output = {
            'index': index,
            "input": input_text,
            "outputs": answer,
            "length": length,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():   
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    icl_example = sys_vartrack_w_noise_random(num_samples=1, 
                                              max_seq_length=500, 
                                              incremental=5,
                                              num_chains=args.num_chains, 
                                              num_hops=args.num_hops)[0]
    write_jsons = sys_vartrack_w_noise_random(num_samples=args.num_samples,
                                              max_seq_length=args.max_seq_length, 
                                              num_chains=args.num_chains,
                                              num_hops=args.num_hops,
                                              icl_example=icl_example)
    
    write_manifest(save_file, write_jsons)

if __name__=="__main__":
    main()