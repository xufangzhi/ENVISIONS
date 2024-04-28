from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/dpo_iter1_dpo_tune_sft_iter0_7B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v17_llama2chat13b_sft_iter9_sft_tune_llama2chat_13B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_symbolllm_instruct_sft_iter0_sft_tune_symbolllm_instruct_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_symbolllm_instruct_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/metamathqa_3w7_dpo_tune_symbolllm_instruct_7B"


model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
device = "cuda:0"
model.to(device)
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
    # model_folder = f"metamathqa_filter_0.9_6k_iter1_dpo_tune_iter1_1_7b_beam{beam}"
    model_folder = f"gsm_math_full_v17_llama2chat13b_sft_iter9_sft_tune_llama2chat_13b_beam{beam}_repaired"
    # model_folder = f"gsm_math_full_gpt-3.5-turbo-0125_7b_beam{beam}_repaired"

    # model_folder = f"ultrafeedback_dpo_tune_symbolllm_instruct_7b_beam{beam}"
    # for dataset in ["svamp","asdiv"]:
    # for dataset in ["gsm","gsmhard",]:
    for dataset in ['math',]:
        # for dataset in ['gsm']:

        test_path = f"test_dataset/{dataset}/{dataset}_test.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)

        with open(f"symbol-llm-v2/verified_results/{dataset}/gsm_math_full_v17_llama2chat13b_iter9.json") as file:
        # with open(f"symbol-llm-v2/verified_results/{dataset}/gsm_math_full_gpt-3.5-turbo.json") as file:
            verified_results = json.load(file)


        result = []

        for i in range(0, len(data_test), batch_size):
            if not verified_results[i]['correctness']:

                instruction = "You are provided with a Python code to solve the given problem. You can either repair and refine it, or simply return the original solution.\n"

                batch_prompts = [instruction + "\nThe question is:\n" + data_test[i+j]['input'] \
                           + "\nThe current Python code is:\n" + extract_code_blocks(verified_results[i+j]['response']) \
                           + "\nThe solution code is:\n"
                           for j in range(batch_size)]

                tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
                batch_input_ids = tokenized_prompts.input_ids.cuda()
                attention_mask = tokenized_prompts.attention_mask.cuda()

                # Generate
                batch_outputs = model.generate(batch_input_ids, max_length=2200, num_beams=beam)
                batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
                batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
                batch_prompts = [prompt for prompt in batch_prompts]
                batch_generations = [
                    output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
                ]

                for j in range(batch_size):
                    result_dict = {}
                    result_dict['id'] = i+j
                    result_dict['response'] = batch_generations[j]
                    # print(batch_generations[j])
                    result.append(result_dict)
                # print("-----")
                # print(data_test[i]['output'])

            else:
                result_dict = {}
                result_dict['id'] = i
                result_dict['response'] = ""
                # print(batch_generations[j])
                result.append(result_dict)


            print("==========", i / len(data_test))
                # response_process = "".join(response.replace("\"","'").split())
                # label = "".join(data_test[i]['output'].replace("\"","'").split())


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