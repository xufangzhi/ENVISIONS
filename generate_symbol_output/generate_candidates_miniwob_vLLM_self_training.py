import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
import argparse
import logging
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, theoremqa_prompt
sys.path.append('ENVISIONS/Synapse')
from synapse.agents.miniwob_seeclick import Agent

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
        default=1,
        help="batchsize for vllm",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="miniwob_v17_llama2chat",
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
        help="base model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot prompting",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true", default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.few_shot and args.base_model == f"llama2chat" and args.model_size == "7B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"  # path to the base LLM
    elif args.few_shot and args.base_model == f"llama2chat" and args.model_size=="13B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"  # path to the base LLM
    else:
        PATH_TO_CONVERTED_WEIGHTS=f"ENVISIONS/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)

    for part in [f"part{args.part_id}"]:

        test_path = f"ENVISIONS/open-instruct/data/miniwob_{part}.json"

        with open(test_path) as file:
            data_test = json.load(file)
        print(len(data_test))

        with open(f"ENVISIONS/Synapse/data/miniwob_observation.json") as file:
            obs_dict = json.load(file)
        # automatically load the new observation
        new_obs_list = []
        if not args.few_shot:
            for i in range(len(data_test)):
                obs = obs_dict[data_test[i]['env_name']][int(data_test[i]['id'])+(args.cur_iter+1)*50]
                new_obs_list.append(obs)
            assert len(new_obs_list)==len(data_test), "obs mismatch"

        result = []
        for i in range(0,len(data_test),args.vllm_batchsize):
            result_dict = {}
            sampling_params = SamplingParams(max_tokens=4000 ,n=5)
            instruction = "You are required to navigate the web. To accomplish the task, use methods in Agent class to generate actions, with the following functions. type(characters: str): Type a string via the keyboard. click_xpath(xpath: str): Click an HTML element with a valid XPath. press(key_type: str): Press a key on the keyboard (enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v). click_option(xpath: str): Click an option HTML element in a list with a valid XPath. movemouse(xpath: str): Move the mouse cursor on an HTML element with a valid XPath.\n"
            if args.few_shot:
                prompts = [instruction + "\n" + data_test[j]['few_shot_prompt'] + "\nThe observation is:\n" + data_test[j]['input']
                            + "\nThe action is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]
            else:
                prompts = [instruction + "\n" + "\nThe observation is:\n" + new_obs_list[j]
                            + "\nThe action is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]

            try:
                outputs = llm.generate(prompts, sampling_params)
            except:
                outputs = []

            if outputs:
                # print(len(prompts))
                # print(len(outputs))
                # print(outputs)
                outputs = outputs[:args.vllm_batchsize]
                assert len(outputs)==args.vllm_batchsize, "error"

                for j, output in enumerate(outputs):
                    response_list = output.outputs
                    if len(response_list)!=5:
                        response_list += [response_list[-1] for _ in range(5-len(response_list))]
                    assert len(response_list) == 5, "output mismatch"
                    for response in response_list:
                        response_text = response.text

                        result_dict = {}
                        response_text = response_text.strip()
                        result_dict['id'] = i+j
                        if args.few_shot:
                            result_dict['question'] = data_test[i+j]['input']
                        else:
                            result_dict['question'] = new_obs_list[i+j]
                        result_dict['response'] = response_text
                        result_dict['target'] = data_test[i+j]['label']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)
                        print(response_text)
                        result.append(result_dict)

            else:
                result_dict = {}
                result_dict['id'] = i
                result_dict['question'] = data_test[i]['input']
                result_dict['response'] = ""
                result_dict['target'] = data_test[i]['label']
                result_dict['logprobs'] = -99999
                result.append(result_dict)
                print("The response is empty")
            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]

            print(f"====={i + j}/{len(data_test)}=====", (i + j) / len(data_test))

        print(len(result), len(data_test))
        assert len(result) == len(data_test)*5, "generated length mismatch"

        if args.few_shot:
            test_result_folder = f"ENVISIONS/score_memory/{args.task_prefix}"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter0.json", 'w') as file:
                json.dump(result, file, indent=4)
        else:
            test_result_folder = f"new_generated_data/"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json", 'w') as file:
                json.dump(result, file, indent=4)

        print(f"[info] the result file has been saved.")
        print("==========")


if __name__ == "__main__":
    main()