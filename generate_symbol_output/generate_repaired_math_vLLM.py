import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, theoremqa_prompt


# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_v2_sft_iter1_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_WEIGHTS="/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v10_sft_iter8_sft_tune_llama2chat_7B/"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/sft_iter0_sft_tune_dpo_iter0_7B"
# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"
# PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
# available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)

# model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
# tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
# device = "cuda:0"
# model.to(device)
batch_size = 1

def extract_code_blocks(text):
    text = text.strip()
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        if code_blocks[0].startswith("python\n"):
            return "```python\n"+code_blocks[0].split("python\n")[1].strip()+"\n```"
        else:
            return "```python\n"+code_blocks[0]+"\n```"
    else:
        return "```python\n"+text+"\n```".strip()

for beam in [1]:
    for part in ["part1"]:
    # for part in ["part20"]:

        # part = "part1"
        # test_path = f"test_dataset/{dataset}/{dataset}_test.json"
        # test_path = f"symbol-llm-v2/open-instruct/data/metamathqa_{part}.json"
        test_path = f"symbol-llm-v2/open-instruct/data/gsm_math_full_{part}.json"
        # test_path = f"symbol-llm-v2/open-instruct/data/theoremqa_train.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)

        # with open(f"new_generated_data/theoremqa_v2_iter2.json") as file:
        with open(f"new_generated_data/gsm_math_full_v10_{part}_iter9.json") as file:
            data_before = json.load(file)

        assert len(data_test)==len(data_before)//5
        result = []
        for i in range(0,len(data_test),1):
            result_dict = {}
            prompt = data_test[i]['input']
            sampling_params = SamplingParams(max_tokens=2800,n=1)
            # prompts = [prompt]
            instruction = "Repair the provided Python code to solve the given problem."

            prompts = [instruction + "\nThe question is:\n" + data_test[i]['input'] \
                        + "\nThe current Python code is:\n" + extract_code_blocks(data_before[i*5+j]['response']) \
                        + "\nThe repaired code is as follows:\n"
                       for j in range(5)]

            outputs = llm.generate(prompts, sampling_params)
            # for _ in range(5):
            if outputs:
                for j, output in enumerate(outputs):
                    response_list = output.outputs
                    for response in response_list:
                        response = response.text
                        # response = llm.generate(prompts, sampling_params)[0].outputs[0].text
                        # generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
                        # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        result_dict = {}
                        # response = response.split(prompt)[1].strip()
                        response = response.strip()
                        result_dict['id'] = i
                        result_dict['question'] = data_test[i]['input']
                        result_dict['response'] = response
                        result_dict['target'] = data_test[i]['label']

                        result.append(result_dict)
                        print(response)
            else:
                result_dict = {}
                # response = response.split(prompt)[1].strip()
                result_dict['id'] = i
                result_dict['question'] = data_test[i]['input']
                result_dict['response'] = ""
                result_dict['target'] = data_test[i]['label']
                result.append(result_dict)
                print("The response is empty")

                    # print("-----")
                    # print(data_test[i]['output'])
            print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))


        test_result_folder = f"new_generated_data/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
        with open(f"{test_result_folder}/gsm_math_full_v10_{part}_iter9_repaired.json", 'w') as file:
        # with open(f"{test_result_folder}/theoremqa_v2_iter2_repaired.json", 'w') as file:
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