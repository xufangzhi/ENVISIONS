from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompt import theoremqa_prompt

# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_13b_instruct"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v10_sft_iter7_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_v2_sft_iter1_sft_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_v6_iter10_all2_by_order_sft_tune_llama2chat_7B"
PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_symbolllm_instruct_sft_tune_llama2chat_7B"


model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
device = "cuda:0"
model.to(device)
batch_size = 1

for beam in [1]:
    # model_folder = f"theoremqa_v2_sft_iter1_sft_tune_llama2chat_7b_beam{beam}"
    # model_folder = f"gsm_math_full_v10_sft_iter7_sft_tune_llama2chat_7b_beam{beam}"
    # model_folder = f"theoremqa_llama2chat_fs_7b_beam{beam}"
    model_folder = f"gsm_math_full_gpt-3.5-turbo-0125_sft_tune_llama2chat_7b_beam{beam}"
    # model_folder = f"gsm_math_full_v8_sft_iter4_sft_tune_dpo_iter4_7b_beam{beam}"

    # model_folder = f"ultrafeedback_dpo_tune_symbolllm_instruct_7b_beam{beam}"
    for dataset in ['theoremqa']:
        # for dataset in ['gsm']:

        test_path = f"symbol-llm-v2/test_dataset/{dataset}/{dataset}_test.json"
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
            instruction = "Write Python code to solve the question.\nThe returned value of the program is supposed to be the answer. \
                It should be integer or float or list of integer/float.\n"

            batch_prompts = [instruction + f"\nThe question is: {data_test[i+j]['input']}\n" + "The solution code is:\n" for j in range(batch_size)]

            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
            batch_input_ids = tokenized_prompts.input_ids.cuda()
            attention_mask = tokenized_prompts.attention_mask.cuda()

            # Generate
            batch_outputs = model.generate(batch_input_ids, max_length=3000, num_beams=beam)
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