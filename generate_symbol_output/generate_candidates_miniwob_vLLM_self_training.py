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
sys.path.append('symbol-llm-v2/Synapse')
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
        default="phi3",
        help="Model size",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="3B",
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

    if args.few_shot and args.base_model=="llemma":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c"
    elif args.few_shot and args.base_model=="symbolllm":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
    elif args.few_shot and args.base_model == f"gpt2xl":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8"
    elif args.few_shot and args.base_model == f"tinyllama":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    elif args.few_shot and args.base_model == f"opt" and args.model_size=="2.7B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254"
    elif args.base_model == f"well":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_7B"
    elif args.few_shot and args.base_model == f"llama2chat" and args.model_size == "7B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
    elif args.few_shot and args.base_model == f"llama2chat" and args.model_size=="13B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"
    elif args.few_shot and args.base_model == f"llama3chat" and args.model_size == "8B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/NLP-A100/NLP-A100_hdd/model/Meta-Llama-3-8B"
    elif args.few_shot and args.base_model == f"phi3" and args.model_size == "3B":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35"
    else:
        PATH_TO_CONVERTED_WEIGHTS=f"/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"
    # PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/sft_iter0_sft_tune_dpo_iter0_7B"
    # PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"
    # PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
    PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
    # available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    # print(available_gpus)
    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)

    # model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    # tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    # device = "cuda:0"
    # model.to(device)
    batch_size = 1

    for beam in [1]:
        for part in [f"part{args.part_id}"]:

            test_path = f"symbol-llm-v2/open-instruct/data/miniwob_{part}.json"

            with open(test_path) as file:
                data_test = json.load(file)
            print(len(data_test))

            with open(f"symbol-llm-v2/Synapse/data/miniwob_observation.json") as file:
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
                # prompt = data_test[i]['input']
                sampling_params = SamplingParams(max_tokens=4000 ,n=5)
                # prompts = [prompt]
                # instruction = "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```\n"
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
                    # for _ in range(5):
                    print(len(prompts))
                    print(len(outputs))
                    # print(outputs)
                    outputs = outputs[:args.vllm_batchsize]
                    assert len(outputs)==args.vllm_batchsize, "1111111"

                    for j, output in enumerate(outputs):
                        response_list = output.outputs
                        if len(response_list)!=5:
                            response_list += [response_list[-1] for _ in range(5-len(response_list))]
                        assert len(response_list) == 5, "output mismatch"
                        for response in response_list:
                            response_text = response.text
                            # response = llm.generate(prompts, sampling_params)[0].outputs[0].text
                            # generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
                            # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            result_dict = {}
                            # response = response.split(prompt)[1].strip()
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
                # except:
                #     print("TypeError: argument 'tokens': 'NoneType' object cannot be converted to 'PyString'")
                #     break

                else:
                    result_dict = {}
                    # response = response.split(prompt)[1].strip()
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
                test_result_folder = f"symbol-llm-v2/score_memory/{args.task_prefix}"
                if not os.path.exists(test_result_folder):
                    os.system(f"mkdir -p {test_result_folder}")
                with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter0.json", 'w') as file:
                    json.dump(result, file, indent=4)
            else:
                test_result_folder = f"new_generated_data/"
                if not os.path.exists(test_result_folder):
                    os.system(f"mkdir -p {test_result_folder}")
                # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
                with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json", 'w') as file:
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


if __name__ == "__main__":
    main()