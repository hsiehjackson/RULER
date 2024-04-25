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
Create a dataset jsonl file for frequent words extraction.

python freq_words_extraction.py   \
    --save_dir=./ \
    --save_name=vt \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --tokens_to_generate 30 \
    --num_samples 10 \
    --random_seed 42  \
    --alpha 2.0 \
    --template "[INST] Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? [/INST] Answer: According to the coded text above, the three most frequently appeared words are:"
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import string
import numpy as np
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 
from tokenizer import select_tokenizer
from scipy.special import zeta 

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, default=50, help='number of tokens to generate')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default='', help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--coded_wordlen", type=int, default=6, help="length of synthetic word")
parser.add_argument("--vocab_size", type=int, default=-1, help='synthetic vocab size to sample from')
parser.add_argument("--alpha", type=float, default=2.0, help='zeta distribution alpha')
parser.add_argument("--add_fewshot", action="store_true", default=False)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

def generate_input_output(max_len, num_words=-1, coded_wordlen=6, vocab_size=2000, incremental=10, alpha=2.0):
    # generate vocab
    vocab = [''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)) for _ in range(vocab_size)]
    while len(set(vocab)) < vocab_size:
        vocab.append(''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(list(set(vocab)))
    random.Random(args.random_seed).shuffle(vocab)
    vocab[0] = '...' # treat the top ranked as noise

    # sample words
    template = args.template
    def gen_text(num_words):
        k = np.arange(1, len(vocab)+1)
        sampled_cnt = num_words*(k**-alpha)/zeta(alpha)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.Random(args.random_seed).shuffle(sampled_words)
        return template.format(context=' '.join(sampled_words), query=''), vocab[1:4]
    
    if num_words > 0:
        num_words = num_words
        text, answer = gen_text(num_words)
        while len(TOKENIZER.text_to_tokens(text)) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen # init
        text, answer = gen_text(num_words)
        while len(TOKENIZER.text_to_tokens(text)) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental
    text, answer = gen_text(num_words)
    return text, answer, num_words

def sys_kwext(num_samples: int, max_seq_length: int, incremental: int = 10):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    vocab_size = max_seq_length // 50 if args.vocab_size == -1 else args.vocab_size

    # get number of words
    input_max_len = max_seq_length 
    _, _, num_example_words = generate_input_output(input_max_len, 
                                                    coded_wordlen=args.coded_wordlen, 
                                                    vocab_size=vocab_size, 
                                                    incremental=input_max_len//32, 
                                                    alpha=args.alpha) 
    print('num_example_words:', num_example_words)
    # Generate samples
    for index in tqdm(range(num_samples)):
        
        # construct input
        input_max_len = max_seq_length 
        input_text, answer, _ = generate_input_output(input_max_len,
                                                   num_words=num_example_words,
                                                   coded_wordlen=args.coded_wordlen, 
                                                   vocab_size=vocab_size,
                                                   incremental=input_max_len//32,
                                                   alpha=args.alpha)
        

        length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate

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
    write_jsons = sys_kwext(num_samples=args.num_samples, max_seq_length=args.max_seq_length, 
                            incremental=10)
    
    write_manifest(save_file, write_jsons)

if __name__=="__main__":
    main()