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

Templates = {
    'base': "{task_template}",

    'meta-chat': "[INST] {task_template} [/INST]",

    'vicuna-chat': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",

    'lwm-chat': "You are a helpful assistant. USER: {task_template} ASSISTANT: ",

    'command-r-chat': "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",

    'chatglm-chat': "[gMASK]sop<|user|> \n {task_template}<|assistant|> \n ",
    
    'RWKV': "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it\n\nUser: {task_template}\n\nAssistant:",

    'Phi3': "<|user|>\n{task_template}<|end|>\n<|assistant|>\n",
}