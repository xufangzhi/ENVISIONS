# Copyright 2022 PAL Authors. All rights reserved.
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

import openai
import time
import os


# openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = "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"
# openai.api_key = "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"
# openai.api_key = "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"
# 120 dollar
# openai.api_key = "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"   # for aqua 5w
# openai.api_key = "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"   # for aqua
# openai.api_key = "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"   # for aqua 6w
# openai.organization = "org-tvf0r4JnjRo3xvSeOVmwaDOP"
# openai.api_key = "sk-WwTFy3etFQZH6uPG9f2a15E0030d49Cb85E7510a65059117"     # aqua 5w
# openai.api_key = "sk-tj75LnrZqLNwtNLs10732a47D4404eAcBaDd545065Ef44Cb"     # aqua 3w
# openai.api_key = "sk-Q6WQ79VqJin0gzSg4f6a3e2092D44aB391F94f65C33b2807"    # aqua
# openai.api_key = "sk-B0v8pGUDa2zwY1xF259582D148914325B667D5E3C6761b53"   # aqua 4w

openai.api_key = "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"
# openai.organization = "org-QeHTBCcuRBiUC5VWJ9mpedsI"
# openai.api_key = "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"
# openai.organization = "org-tvf0r4JnjRo3xvSeOVmwaDOP"
# GPT-3 API
def call_gpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
    
        
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            if model.startswith('gpt-4') or model.startswith('gpt-3.5-turbo'):
                ans = chat_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions)
            else:
                ans = completions_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions)
            completions.extend(ans)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')

def completions_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        n=n,
        best_of=best_of)
    return [choice['text'] for choice in ans['choices']]

def chat_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.ChatCompletion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
            {'role': 'user', 'content': prompt}],
        temperature=temperature,
        top_p=top_p,
        n=n)
    return [choice['message']['content'] for choice in ans['choices']]


def call_chat_gpt(messages, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=1.0, max_tokens=128):
    wait = 1
    while True:
        try:
            ans = openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                stop=stop,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            return ans.choices[0]['message']['content']
        except openai.error.RateLimitError as e:
            time.sleep(min(wait, 60))
            wait *= 2
    raise RuntimeError('Failed to call chat gpt')