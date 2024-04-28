from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, theoremqa_prompt

# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_sft_tune_llemma_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_star_sft_iter9_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/logic_v17_llama2chat_sft_iter9_sft_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_symbolllm_instruct_sft_iter0_sft_tune_symbolllm_instruct_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_symbolllm_instruct_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"


model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
device = "cuda:0"
model.to(device)
batch_size = 1

for beam in [2]:
    # model_folder = f"symbol-llm_7b_beam{beam}"
    # model_folder = f"gsm_math_full_gpt-3.5-turbo-0125_sft_tune_llemma_7b_beam{beam}"
    model_folder = f"logic_v17_llama2chat_sft_iter9_sft_tune_llama2chat_7b_beam{beam}"

    for dataset in ['prontoqa']:

        test_path = f"test_dataset/{dataset}/{dataset}_dev.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []


        for i in range(0, len(data_test), batch_size):
            instruction = "Generate the logical representation for the given context and question."
            batch_prompts = [instruction + f"\nThe context is: {data_test[i+j]['context']}\n" + f"\nThe question is: {data_test[i+j]['question']}\n" + \
                             "The logical representation is:\n" for j in range(batch_size)]

            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
            batch_input_ids = tokenized_prompts.input_ids.cuda()
            attention_mask = tokenized_prompts.attention_mask.cuda()

            # Generate
            batch_outputs = model.generate(batch_input_ids, max_length=4096, num_beams=beam)
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