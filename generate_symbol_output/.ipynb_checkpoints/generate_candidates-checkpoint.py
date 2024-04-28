from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from prompt import math_prompt


PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_base"
PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_base"


model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
device = "cuda:0"
model.to(device)

for beam in [1]:
    model_folder = f"candidates_7b_beam{beam}"
    part = "part9"
    # test_path = f"test_dataset/{dataset}/{dataset}_test.json"
    test_path = f"./symbol-llm-v2/open-instruct/data/math_origin_{part}.json"
    with open(test_path) as file:
        data_test = json.load(file)
    print(test_path)
    result = []
    for i in range(len(data_test)):
        instruction = "Write Python code to solve the question.\n"

        prompt = instruction + f"\nThe question is: {data_test[i]['input']}\n" + "The solution code is:\n"
        # prompt = instruction + f"\nThe question is: {data_test[i]['input']}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        for _ in range(5):
            generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
            response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result_dict = {}
            response = response.split(prompt)[1].strip()
            result_dict['id'] = i
            result_dict['question'] = data_test[i]['input']
            result_dict['response'] = response

            result.append(result_dict)
            print(response)
            # print("-----")
            # print(data_test[i]['output'])
            print(f"====={i}/{len(data_test)}=====", i/len(data_test))

    test_result_folder = f"new_generated_data/"
    if not os.path.exists(test_result_folder):
        os.system(f"mkdir -p {test_result_folder}")
    with open(f"{test_result_folder}/math_new_{part}.json",'w') as file:
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