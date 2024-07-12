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
from prompt import math_prompt, theoremqa_prompt, proofwriter_prompt, foliolr_prompt

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
        default=6,
        help="Original datasets are split into several parts, this argument returns the part index",
    )
    parser.add_argument(
        "--vllm_batchsize",
        type=int,
        default=1,
        help="batchsize for vllm",
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.few_shot and args.base_model == f"llama2chat" and args.model_size=="7B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"  # path to the base LLM
    elif args.few_shot and args.base_model == f"llama2chat" and args.model_size=="13B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"  # path to the base LLM
    else:
        PATH_TO_CONVERTED_WEIGHTS=f"ENVISIONS/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)

    for part in [f"part{args.part_id}"]:
        test_path = f"ENVISIONS/open-instruct/data/proofwriter_{part}.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []
        for i in range(0,len(data_test),args.vllm_batchsize):
            result_dict = {}

            sampling_params = SamplingParams(max_tokens=3500,n=5)
            instruction = "Generate the logical representation for the given context and question. "

            if args.few_shot:
                prompts = [instruction + "\n" + proofwriter_prompt.PROOFWRITER_PROMPT + "\nThe context is:\n" + data_test[j]['context']
                            + "\nThe question is:\n" + data_test[j]['question'] + "\nThe logical representation is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]
            else:
                prompts = [instruction + "\n" + "\nThe context is:\n" + data_test[j]['context'] + "\nThe question is:\n" \
                            + data_test[j]['question'] + "\nThe logical representation is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]
            try:
                outputs = llm.generate(prompts, sampling_params)
            except:
                print("error")
                prompts = []
                outputs = []

            if outputs:
                # for _ in range(5):
                outputs = outputs[:args.vllm_batchsize]
                print(len(outputs))
                for j, output in enumerate(outputs):
                    response_list = output.outputs
                    print(len(response_list))
                    if len(response_list) != 5:
                        response_list += [response_list[-1] for _ in range(5 - len(response_list))]
                    assert len(response_list) == 5, "output mismatch"
                    for response in response_list:
                        response_text = response.text
                        result_dict = {}
                        response_text = response_text.strip()
                        result_dict['id'] = i+j
                        result_dict['context'] = data_test[i+j]['context']
                        result_dict['question'] = data_test[i+j]['question']
                        result_dict['response'] = response_text
                        result_dict['label'] = data_test[i+j]['answer']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)

                        result.append(result_dict)

            else:
                result_dict = {}
                result_dict['id'] = i
                result_dict['context'] = data_test[i]['context']
                result_dict['question'] = data_test[i]['question']
                result_dict['response'] = ""
                result_dict['label'] = data_test[i]['answer']
                result_dict['logprobs'] = -99999
                result.append(result_dict)
                print("The response is empty")

            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]

            print(f"====={i + j}/{len(data_test)}=====", (i + j) / len(data_test))


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