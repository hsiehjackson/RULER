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
Add a new task:

TASK_NAME: {
    'metric_fn': the metric function with input (predictions: [str], references: [[str]]) to compute score.
}
"""


def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)
    

TASKS = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
