import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from prompt import math_prompt


PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"

# available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
# llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
device = "cuda:0"
model.to(device)
batch_size = 1

for beam in [1]:
    for part in ["part5"]:
        # part = "part1"
        # test_path = f"test_dataset/{dataset}/{dataset}_test.json"
        test_path = f"symbol-llm-v2/open-instruct/data/new_math_preference_data_{part}.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []
        for i in range(0, len(data_test), batch_size):
            for _ in range(5):
                # instruction = "Write Python code to solve the question.\n"
                batch_prompts = [data_test[i+j]['input'] for j in range(batch_size)]

                torch.cuda.empty_cache()
                with torch.no_grad():
                    tokenized_prompts = tokenizer(batch_prompts, padding=True, return_tensors="pt")
                    batch_input_ids = tokenized_prompts.input_ids.cuda()
                    attention_mask = tokenized_prompts.attention_mask.cuda()

                    # Generate
                    batch_outputs = model.generate(batch_input_ids, max_length=1200, num_beams=beam, do_sample=True)
                    batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
                    batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
                    batch_prompts = [prompt for prompt in batch_prompts]
                    batch_generations = [
                        output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
                    ]
                    # print(batch_generations)
                for j in range(batch_size):
                    result_dict = {}
                    result_dict['id'] = i + j
                    result_dict['response'] = batch_generations[j]
                    # print(batch_generations[j])
                    result.append(result_dict)
                # print("-----")
                # print(data_test[i]['output'])
                print("==========", i / len(data_test))

        test_result_folder = f"new_generated_data/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        with open(f"{test_result_folder}/math_add_{part}_no_vLLM_symbolllm_instruct.json",'w') as file:
            json.dump(result, file, indent=4)

        # with open(f"{test_result_folder}/prediction.txt",'w') as file:
        #     for i in range(len(result)):
        #         if result[i]['response']=="":
        #             file.write("wrong")
        #         else:
        #             file.write(result[i]['response'])
        #         file.write('\n')
        print(f"[info] the result file has been saved.")
        print("==========")