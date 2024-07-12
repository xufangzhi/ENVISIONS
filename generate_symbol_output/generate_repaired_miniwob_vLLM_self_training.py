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
        default="agent",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=0,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--vllm_batchsize",
        type=int,
        default=4,
        help="batchsize for vllm",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="miniwob_llama2chat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        default=8,
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


def extract_action_blocks(text):
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return text.strip()


def main():
    args = parse_args()

    PATH_TO_CONVERTED_WEIGHTS = f"ENVISIONS/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)

    for part in [f"part{args.part_id}"]:

        test_path = f"ENVISIONS/open-instruct/data/miniwob_{part}.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)

        with open(f"new_generated_data/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json") as file:
            data_before = json.load(file)

        assert len(data_test)==len(data_before)//5
        result = []
        for i in range(0,len(data_test),1):
            result_dict = {}

            sampling_params = SamplingParams(max_tokens=4000,n=1)

            instruction = "You are required to navigate the web. To accomplish the task, use methods in Agent class to generate actions, with the following functions. type(characters: str): Type a string via the keyboard. click_xpath(xpath: str): Click an HTML element with a valid XPath. press(key_type: str): Press a key on the keyboard (enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v). click_option(xpath: str): Click an option HTML element in a list with a valid XPath. movemouse(xpath: str): Move the mouse cursor on an HTML element with a valid XPath.\n"
            instruction += "You should repair the current action to finish the task."
            prompts = [instruction + "\nThe observation is:\n" + data_before[i*5+j]['question'] \
                        + "\nThe current action is:\n" + extract_action_blocks(data_before[i*5+j]['response']) \
                        + "\nThe repaired action is:\n"
                        for j in range(5)]

            outputs = llm.generate(prompts, sampling_params)
            if outputs:
                # print(len(outputs))
                outputs = outputs[:5]   # trunct to 5
                for j, output in enumerate(outputs):
                    # print(output)
                    response_list = output.outputs
                    print(len(response_list))

                    for response in response_list:
                        response_text = response.text

                        result_dict = {}
                        response_text = response_text.strip()
                        result_dict['id'] = i
                        result_dict['question'] = data_before[i*5+j]['question']
                        result_dict['response'] = response_text
                        result_dict['target'] = data_test[i]['label']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)
                        result.append(result_dict)
                        # print(response)
            else:
                result_dict = {}
                result_dict['id'] = i
                result_dict['question'] = data_before[i*5+j]['question']
                result_dict['response'] = ""
                result_dict['target'] = data_test[i]['label']
                result_dict['logprobs'] = -99999
                result.append(result_dict)
                print("The response is empty")


            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]
            assert len(result)==(i+1)*5, "1111"
            print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))


        test_result_folder = f"new_generated_data/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
        with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}_repaired.json", 'w') as file:
        # with open(f"{test_result_folder}/theoremqa_v2_iter2_repaired.json", 'w') as file:
            json.dump(result, file, indent=4)

        print(f"[info] the result file has been saved.")
        print("==========")

if __name__ == "__main__":
    main()