from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, theoremqa_prompt

# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_claude_sft_tune_deepseekchat_7B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_star_sft_iter9_sft_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v17_sft_iter8_sft_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v17_deepseekchat_sft_iter9_sft_tune_deepseekchat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v17_deepseekchat_sft_iter7_sft_tune_deepseekchat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/model/Meta-Llama-3-8B"
PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"

model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
# tokenizer.pad_token = tokenizer.eos_token
# model.generation_config = GenerationConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
# print(tokenizer.eos_token)
device = "cuda:0"
model.to(device)
batch_size = 1

for beam in [2]:
    # model_folder = f"metamathqa_filter_0.9_6k_iter1_dpo_tune_iter1_1_7b_beam{beam}"
    model_folder = f"gsm_math_full_star_sft_iter9_sft_tune_llama2chat_7b_beam{beam}"
    # model_folder = f"gsm_math_full_v17_sft_iter8_sft_tune_llama2chat_7b_beam{beam}"
    # model_folder = f"phi3_3b_fs_beam{beam}"
    # model_folder = f"gsm_math_full_v17_deepseekchat_sft_iter9_sft_tune_deepseekchat_7b_beam{beam}"
    # model_folder = f"ultrafeedback_dpo_tune_symbolllm_instruct_7b_beam{beam}"
    for dataset in ["math", "gsm","gsmhard","svamp","asdiv"]:
    # for dataset in ["svamp","asdiv",]:
    # for dataset in ['gsm','gsmhard']:
    # for dataset in ['math']:

        test_path = f"test_dataset/{dataset}/{dataset}_test.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []
        # for i in range(len(data_test)):
        #     result_dict = {}
        #     instruction = "Write Python code to solve the question.\n"
        #
        #     prompt = instruction + f"\nThe question is: {data_test[i]['input']}\n" + "The solution code is:\n"
        #     # prompt = instruction + math_prompt.MATH_PROMPT_FS_TEST + f"Question: {data_test[i]['input']}\n"
        #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
        #
        #     # Generate
        #     generate_ids = model.generate(inputs.input_ids, max_length=2000, num_beams=beam)
        #     response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #     response = response.split(prompt)[1].strip()
        #     result_dict['id'] = i
        #     result_dict['response'] = response
        #     # result_dict['label'] = data_test[i]['output']
        #     result.append(result_dict)
        #     print(response)
        #     # print("-----")
        #     # print(data_test[i]['output'])
        #     print("==========", i / len(data_test))
        #     # response_process = "".join(response.replace("\"","'").split())
        #     # label = "".join(data_test[i]['output'].replace("\"","'").split())


        for i in range(0, len(data_test), batch_size):
            instruction = "Write Python code to solve the question.\n"
            batch_prompts = [instruction + f"\nThe question is: {data_test[i+j]['input']}\n" + "The solution code is:\n" for j in range(batch_size)]

            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
            batch_input_ids = tokenized_prompts.input_ids.cuda()
            attention_mask = tokenized_prompts.attention_mask.cuda()

            # Generate
            batch_outputs = model.generate(batch_input_ids, max_length=2048, num_beams=beam)
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
                print(batch_generations[j])
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