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

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
#     131072
#     65536
#     32768
#     16384
    8192
    4096
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        llama2-7b-chat)
            MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="sglang"
            ;;
        gpt-3.5-turbo)
            MODEL_PATH="gpt-3.5-turbo-0125"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gpt-4-turbo)
            MODEL_PATH="gpt-4"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gemini_1.0_pro)
            MODEL_PATH="gemini-1.0-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
        gemini_1.5_pro)
            MODEL_PATH="gemini-1.5-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
        mistral-7b-instruct-v0.2)
            MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="sglang"
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
