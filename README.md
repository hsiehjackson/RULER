# 📏 RULER: What’s the Real Context Size of Your Long-Context Language Models?

This repository contains code for our paper [RULER: What’s the Real Context Size of Your Long-Context Language Models](https://arxiv.org/abs/2404.06654). RULER generates synthetic examples to evaluate long-context language models with configurable sequence length and task complexity. We benchmark 17 open-source models across 4 task categories (in total 13 tasks) in RULER, evaluating long-context capabilities beyond simple in-context recall. Here are our main results.

|Models|Claimed Length|Effective Length|4K|8K|16K|32K|64K|128K|Avg.|wAvg. (inc)|wAvg. (dec)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (7B)|4K||85.6|
[Gemini-1.5-pro](https://ai.google.dev/gemini-api/docs/models/gemini#:~:text=Gemini-,Gemini%201.5%20Pro%20(Preview%20only),-Text%20and%20images)|1M|>128K|<ins>96.7</ins>|<ins>95.8</ins>| <ins>96.0</ins>| <ins>95.9</ins>|<ins>95.9</ins>|<ins>94.4</ins>|95.8|95.5 (1st)|96.1 (1st)|
[GPT-4-1106-preview](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4#:~:text=gpt%2D4%2D1106%2Dpreview,Up%20to%20Apr%202023)|128K|64K|<ins>96.6</ins>|<ins>96.3</ins>| <ins>95.2</ins>| <ins>93.2</ins>|<ins>87.0</ins>|81.2|91.6|89.0 (2nd)|94.1 (2nd)|
[Command-R-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) (104B)|128K|32K|<ins>95.6</ins>| <ins>95.2</ins>| <ins>94.2</ins>|<ins>92.0</ins>|84.3|63.1|87.4|82.7 **(7th)**|92.1 **(3rd)**|
[GLM-4-chat](THUDM/glm-4-9b-chat-1m) (9B)|1M|64K|<ins>94.7</ins>|<ins>92.8</ins>|<ins>92.1</ins>|<ins>89.9</ins>|<ins>86.7</ins>|83.1|89.9|88.0 **(3rd)**|91.7 **(4th)**|
[GradientAI/Llama3*](https://huggingface.co/gradientai/Llama-3-70B-Instruct-Gradient-1048k)(70B)|1M|32K|<ins>95.2</ins>|<ins>93.4</ins>|<ins>93.4</ins>|<ins>89.4</ins>|82.6|72.0|87.7|84.0	**(6th)**|91.3	**(5th)**|
[Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01) (35B)|128K|32K| <ins>93.8</ins>| <ins>93.3</ins>| <ins>92.4</ins>|<ins>89.5</ins>|84.9|76.0|88.3|85.5 **(4th)**|91.1 **(6th)**|
[Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-insruct-v0.1) (39B/141B)|64K|32K| <ins>95.6</ins>| <ins>94.9</ins>| <ins>93.4</ins>|<ins>90.9</ins>|84.7|31.7|81.9|73.5 **(8th)**|90.3 **(7th)**|
[Yi](https://huggingface.co/01-ai/Yi-34B-200K) (34B)|200K|32K| <ins>93.3</ins>| <ins>92.2</ins>| <ins>91.3</ins>|<ins>87.5</ins>|83.2|77.3|87.5|84.8 **(5th)**|90.1 **(8th)**|
[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-insruct-v0.1) (12.9B/46.7B)|32K|32K| <ins>94.9</ins>| <ins>92.1</ins>| <ins>92.5</ins>|<ins>85.9</ins>|72.4|44.5|80.4|72.8 (9th)|87.9 (9th)|
[FILM-7B*](https://arxiv.org/pdf/2404.16811) (7B)|32K|32K|<ins>92.8</ins>|<ins>88.2</ins>|<ins>88.1</ins>|<ins>86.9</ins>|70.1|27.1|75.5|	66.4 **(11th)**|84.7 **(10th)**|
[Meta/Llama3*](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (RoPE $\theta$=16M)(70B)|8K|>8K|<ins>95.4</ins>|<ins>94.7</ins>|<ins>93.2</ins>|<ins>85.9</ins>|22.5|0.0|65.3|48.6	**(14th)**|82.0	**(11th)**|
[Mistral](https://huggingface.co/mistralai/Mistral-7B-insruct-v0.2) (7B)|32K|16K| <ins>93.6</ins>| <ins>91.2</ins>|<ins>87.2</ins>|75.4|49.0|13.8|68.4|55.6 **(15th)**|81.2 **(12th)**|
[ChatGLM](https://huggingface.co/THUDM/chatglm3-6b-128K) (6B)|128K|4K|<ins>87.8</ins>|83.4|78.6|69.9|56.0|42.0|69.6|62.0 (13th)|77.2 (13th)|
[LWM](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M) (7B)|1M|<4K|82.3|78.4|73.7|69.1|68.1|65.0|72.8|69.9 **(10th)**|75.7 **(14th)**|
[Phi3](https://huggingface.co/microsoft/Phi-3-mini-128K-instruct) (3.8B)|128K|4K|<ins>86.7</ins>|78.1|75.6|70.3|58.9|43.3|68.8|62.2 **(12th)**|75.5 **(15th)**|
[DBRX](https://huggingface.co/databricKs/dbrx-insruct) (36B/132B)|32K|8K|<ins>95.1</ins>|<ins>93.8</ins>|83.6|63.1|2.4|0.0|56.3|38.0 (16th)|74.7 (16th)|
[Qwen](https://huggingface.co/Qwen/Qwen1.5-72B-Chat) (72B)|32K|8K|<ins>94.9</ins>|<ins>93.8</ins>|78.0|67.8|0.0|0.0|55.7|37.5 (17th)|74.0 (17th)|
[Together](https://huggingface.co/togethercomputer/Llama-2-7B-32K-insruct) (7B)|32K|4K|<ins>88.2</ins>|81.1|69.4|63.0|0.0|0.0|50.3|33.8 (18th)|66.7 (18th)|
[LongChat](https://huggingface.co/lmsys/longchat-7b-v1.5-32K) (7B)|32K|<4K|84.7|79.9|70.8|59.3|0.0|0.0|49.1|33.1 (19th)|65.2 (19th)|
[LongAlpaca](https://huggingface.co/YuKang/LongAlpaca-13B) (13B)| 32K|<4K|60.6|57.0|56.6|43.6|0.0|0.0|36.3|24.7 (20th)|47.9 (20th)|


- Despite achieving nearly perfect performance on the vanilla needle-in-a-haystack (NIAH) test, all models (except for Gemini-1.5-pro) exhibit large degradation on tasks in RULER as sequence length increases. 
- While all models claim context size of 32k tokens or greater (except for Llama3), only half of them can effectively handle sequence length of 32K by exceeding a qualitative threshold, Llama2-7b performance at 4K (85.6%). The performance exceeding the threshold is <ins>underlined</ins>.
- Almost all models fall below the threshold before reaching the claimed context lengths. 
- Notes (Meta/Llama3)
    - The results are evaluated by changing `rope_theta` to 16M in [here](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json#L21).
- Notes (FILM-7B)
    - The results are submitted by authors of this [paper](https://arxiv.org/pdf/2404.16811). They use [YaRN](https://arxiv.org/pdf/2309.00071) without further training for the evaluation length exceeding 32K (64K and 128K). 
    - They do not use the one-shot example for the CWE task.
- Notes (GradientAI/Llama3)
    - The results are submitted by authors. 

## 💡 Requirements

- Docker container: `docker pull cphsieh/ruler:0.1.0`
- The requirements are listed in `docker/Dockerfile` and `docker/requirements.txt`. Use the following command to build the container based on NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:23.08-py3`.
```
cd docker/
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t cphsieh/ruler:0.1.0 .
```


## 🔍 Evaluate long-context LMs
### 1. Download data
- Paul Graham Essays for NIAH are downloaded from [NIAH Github](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays) and [Paul Graham Blog](https://paulgraham.com/articles.html). 
- QA datasets are downloaded from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [HotpotQA](https://hotpotqa.github.io/).
```
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
```
### 2. Download model 
- We download the models from [Huggingface](https://huggingface.co/models).
- The input template of each model is stored in `scripts/data/template.py`. Please add new model template if your new model uses a different chat template.
- (Optional) If you are using [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main), please build your model engine based on their example scripts (e.g., [Llama](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)) with their [Docker container](https://github.com/NVIDIA/TensorRT-LLM/tree/main?tab=readme-ov-file#installation).

### 3. Run evaluation pipeline

- **Setup `run.sh`**
```
GPUS="" # number of GPUs
ROOT_DIR="" # the path that stores generated task samples and model predictions. 
MODEL_DIR="" # the path that contains individual model folders from Huggingface.
ENGINE_DIR="" # the path that contains individual engine folders from TensorRT-LLM.
```
- **Setup `config_models.sh`**
```    
case $MODEL_NAME in
    YOUR_HF_MODEL_NAME)
        MODEL_PATH=${MODEL_DIR}/YOUR_MODEL_FOLDER
        MODEL_TEMPLATE_TYPE="" # base, meta-chat, etc. defined in `scripts/data/template.py`
        MODEL_FRAMEWORK="" # hf or vllm
        ;;
    YOUR_TRTLLM_ENGINE_NAME)
        MODEL_PATH=${ENGINE_DIR}/YOUR_ENGINE_FOLDER
        MODEL_TEMPLATE_TYPE="" # base, meta-chat, etc. defined in `scripts/data/template.py`
        MODEL_FRAMEWORK="trtllm"
        ;;
    YOUR_OPENAI_MODEL_NAME)
        MODEL_PATH="" # OpenAI model name listed in https://platform.openai.com/docs/models/
        MODEL_TEMPLATE_TYPE="base"
        MODEL_FRAMEWORK="openai"
        TOKENIZER_PATH="cl100k_base"
        TOKENIZER_TYPE="openai"
        OPENAI_API_KEY="" # your OpenAI API key
        ;;
    YOUR_GEMINI_MODEL_NAME)
        MODEL_PATH="" # Gemini model name listed in https://ai.google.dev/gemini-api/docs/models/gemini
        MODEL_TEMPLATE_TYPE="base"
        MODEL_FRAMEWORK="gemini"
        TOKENIZER_PATH=$MODEL_PATH
        TOKENIZER_TYPE="gemini"
        GEMINI_API_KEY="" # your Gemini API key
        ;;
```

- **Start evaluation based on our default `synthetic` benchmark**
```
bash run.sh YOUR_MODEL_NAME synthetic
```

## 🧠 (Optional) Customize task complexity 
The tasks to be evaluated on are stored in `scripts/config_tasks.sh`. Configuration of each task is defined in `scripts/synthetic.yaml`. The complexity of each task can be configured by changing the arguments which we describe in detail below.

| Category           |Task name                 | Configurations   |
|:--------------------:|:---------------------------:|--------------------|
| Retrieval          | niah                      |**type_haystack**: `repeat/essay/needle`<br># repeat: repeated noise sentences<br># essay: Paul Graham Essays<br># needle: distracted needles<br><br>**type_needle_k**: `words/numbers/uuids`<br>**type_needle_v**: `words/numbers/uuids`<br># words: adjective-noun<br># numbers: 7 digits<br># uuids: 32 digits<br><br>**num_needle_k**: `int >= 1`<br># add multiple needles in haystack <br>**num_needle_v**: `int >= 1`<br> # retrieve multiple values from a single key<br>**num_needle_q**: `int >= 1`<br> # retrieve multiple values from multiple keys  |
| Multi-hop<br>Tracing  | variable_tracking         | **num_chains**: `int >= 1`<br>#  number of variable name-binding chains<br>**num_hops**: `int >= 1`<br># number of times binding variable names in each chain                    |
| Aggregation        | common_words_extraction   |**freq_cw**: `int >= 1`<br># frequency of common words<br>**freq_ucw**: `int >= 1`<br># frequency of uncommon words<br>**num_cw**: `int >= 1` <br># number of common words                  |
|  Aggregation   | freq_words_extraction     |**alpha**: `float > 1.0`<br># parameter of the distributation to draw synthetic words. Reducing alpha to increase the difficulty of this task. Note that increasing the number of words to return also increases the difficulty of this task, we use `3` in our evaluations as models show worse performance at short context size when more words need to be returned.                    |
| Question<br>Answering | qa                  |**dataset**: `squad` or `hotpotqa`<br># the short-context qa dataset we use



## 🚀 (Optional) Contribute a new synthetic task 
### 1. Create a python script for data preparation
* Add basic arguments (required) and complexity configurations in the python script.
* Verify the script is reproducible given a tokenizer, a sequence length, and a random seed.
* Save the script under the folder `scripts/data/synthetic`.

### 2. Add task template 
* Add `template` and `tokens_to_generate` in `scripts/data/synthetic/constants.py`.
* Add `answer_predfix` to prevent model from refusing to answer.

### 3. Add evaluation metric
* Add the automatic metric to evaluate your task in `scripts/eval/synthetic/constants.py`

### 4. Add required configurations
* Define your task name and complexity configurations in `scripts/synthetic.yaml`.
* Add your task name in `scripts/config_tasks.sh`

## 🛠️ Limitations
While tasks in RULER are designed to be configurable, we only evaluate the above models with 13 task configurations. These tasks were selected because most models can achieve good (some almost perfect) performance at short context size (<= 4K), which leaves ample room to observe degradation as we extend the input length. We did not include more complexed tasks in RULER that models show worse performance at short context size. We also did not stress test every model with more difficult task configurations. Although RULER covers four task categories extending previous evaluation protocol and provides a clean test bed for sanity-checking LMs with known upper bound performance, it is by no means comprehensive enough and it cannot replace the more preferred realistic tasks. We welcome people to contribute new tasks and/or new task categories to help evaluate long-context capabilities. 


## 📝 Citation
```
@article{hsieh2024ruler,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Cheng-Ping Hsieh and Simeng Sun and Samuel Kriman and Shantanu Acharya and Dima Rekesh and Fei Jia and Yang Zhang and Boris Ginsburg},
  year={2024}
  journal={arXiv preprint arXiv:2404.06654},
}
```
Disclaimer: This project is strictly for research purposes, and not an official product from NVIDIA.
