from transformers import AutoTokenizer, LlamaForCausalLM
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from prompt import math_prompt

# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-4-0125-preview_sft_tune_deepseekchat_7B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v17_llama3chat_sft_iter4_sft_tune_llama3chat_8B"
PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/math_dpo_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/4070c4fcff9de9846893cc50429bd008529ff9c6"
# PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"

# model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
# tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
# device = "cuda:0"
# model.to(device)
llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)
tokenizer = llm.get_tokenizer()

for beam in [1]:
    # model_folder = f"gsm_math_full_gpt-4-0125-preview_sft_tune_deepseekchat_7b_beam{beam}"
    model_folder = f"test"
    # for dataset in ['gsm','gsmhard']:
    for dataset in ['svamp',]:
    # for dataset in ['math']:
        test_path = f"test_dataset/{dataset}/{dataset}_test.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []
        for i in range(len(data_test)):
            result_dict = {}
            instruction = "Write Python code to solve the question.\n"

            prompt = instruction + f"\nThe question is: {data_test[i]['input']}\n" + "The solution code is:\n"
            sampling_params = SamplingParams(max_tokens=2048, use_beam_search=False, best_of=beam)
            prompts = [prompt]

            for _ in range(1):
                response = llm.generate(prompts, sampling_params)[0].outputs[0].text
                # generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
                # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                result_dict = {}
                # response = response.split(prompt)[1].strip()
                response = response.strip()
                result_dict['id'] = i
                result_dict['question'] = data_test[i]['input']
                result_dict['response'] = response

                result.append(result_dict)
                print(response)
                # print("-----")
                # print(data_test[i]['output'])
                print(f"====={i}/{len(data_test)}=====", i / len(data_test))
            # result_dict = {}
            # # response = response.split(prompt)[1].strip()
            # response = response.strip()
            # result_dict['id'] = i
            # result_dict['question'] = data_test[i]['input']
            # result_dict['response'] = response
            # result.append(result_dict)
            # print(response)

            # print("==========", i / len(data_test))

        test_result_folder = f"symbol-llm-v2/test_results/{model_folder}/{dataset}/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        with open(f"{test_result_folder}/result.json", 'w') as file:
            json.dump(result, file, indent=4)

        with open(f"{test_result_folder}/prediction.txt", 'w') as file:
            for i in range(len(result)):
                if result[i]['response'] == "":
                    file.write("wrong")
                else:
                    file.write(result[i]['response'])
                file.write('\n')
        print(f"[info] the result file of {dataset} has been saved.")
        print("==========")