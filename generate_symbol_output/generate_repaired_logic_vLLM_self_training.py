import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, theoremqa_prompt


logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="logic",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=0,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="logic_llama2chat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        default=1,
        help="Original datasets are split into several parts, this argument returns the part index",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llama2chat",
        help="Base Model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    args = parser.parse_args()
    return args


def extract_ans(text):
    query_index = text.find("Query:")
    if query_index != -1:
        position = text[query_index:].find("\n\n\n")

        if position != -1:
            return text[:query_index + position].strip().replace("\n\n\n","\n").replace("\n\n","\n").strip()
    return text.strip().replace("\n\n\n","\n").replace("\n\n","\n").strip()


def main():
    args = parse_args()

    PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)


    for part in [f"part{args.part_id}"]:
        test_path = f"symbol-llm-v2/open-instruct/data/proofwriter_{part}.json"

        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)

        with open(f"new_generated_data/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json") as file:
            data_before = json.load(file)

        assert len(data_test)==len(data_before)//5
        result = []
        for i in range(0,len(data_test),1):
            result_dict = {}
            sampling_params = SamplingParams(max_tokens=3500,n=1)

            instruction = "You are provided with a logical representation to solve the given problem. You can either repair and refine it, or you can simply return the original one.\n"
            prompts = [instruction + "\nThe context is:\n" + data_test[i]['context'] + "\nThe question is:\n" + data_test[i]['question'] \
                        + "\nThe current logical representation is:\n" + extract_ans(data_before[i*5+j]['response']) \
                        + "\nThe logical representation is:\n" for j in range(5)]

            try:
                outputs = llm.generate(prompts, sampling_params)
            except:
                outputs = []
            # for _ in range(5):
            if outputs:
                outputs = outputs[:5]  # trunct to 5
                for j, output in enumerate(outputs):
                    # print(output)
                    response_list = output.outputs
                    for response in response_list:
                        response_text = response.text
                        result_dict = {}
                        # response = response.split(prompt)[1].strip()
                        response_text = response_text.strip()
                        result_dict['id'] = i
                        result_dict['question'] = data_test[i]['question']
                        result_dict['response'] = response_text
                        result_dict['label'] = data_test[i]['answer']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)
                        result.append(result_dict)
                        # print(response)
            else:
                result_dict = {}
                # response = response.split(prompt)[1].strip()
                result_dict['id'] = i
                result_dict['question'] = data_test[i]['question']
                result_dict['response'] = ""
                result_dict['label'] = data_test[i]['answer']
                result_dict['logprobs'] = -99999
                result.append(result_dict)
                print("The response is empty")

            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]
            print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))


        test_result_folder = f"new_generated_data/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}_repaired.json", 'w') as file:
            json.dump(result, file, indent=4)

        print(f"[info] the result file has been saved.")
        print("==========")

if __name__ == "__main__":
    main()