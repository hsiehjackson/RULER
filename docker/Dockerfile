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

FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace/

COPY ./requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable \
    && pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
    
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')"]

    