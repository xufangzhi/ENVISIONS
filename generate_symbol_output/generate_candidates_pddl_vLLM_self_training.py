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

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="pddl",
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
        default="logic_v17",
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
        help="Model size",
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.part_id-1)
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # for i in range(torch.cuda.device_count()):
    #     print(torch.cuda.get_device_name(i))
    # torch.cuda.get_device_properties(i)
    # import torch
    # torch.cuda.set_device(args.part_id-1)
    # PATH_TO_CONVERTED_WEIGHTS="/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_v2_sft_iter1_sft_tune_llama2chat_7B"

    # print(args.few_shot, args.base_model)
    if args.few_shot and args.base_model=="llemma":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--EleutherAI--llemma_7b/snapshots/e223eee41c53449e6ea6548c9b71c50865e4a85c"
    elif args.few_shot and args.base_model=="symbolllm":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
    elif args.few_shot and args.base_model == f"mammoth":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--TIGER-Lab--MAmmoTH-Coder-7B/snapshots/857751d2f6255ec1bf5e76bac24cbc24a67288ae"
    elif args.base_model == f"well":
        PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/gsm_math_full_gpt-3.5-turbo-0125_7B"
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
        # for part in ["part20"]:
            # part = "part1"
            # test_path = f"test_dataset/{dataset}/{dataset}_test.json"
            # test_path = f"symbol-llm-v2/open-instruct/data/metamathqa_{part}.json"
            test_path = f"symbol-llm-v2/open-instruct/data/gsm_math_full_{part}.json"
            # test_path = f"symbol-llm-v2/open-instruct/data/theoremqa_train.json"
            with open(test_path) as file:
                data_test = json.load(file)
            print(test_path)
            result = []
            for i in range(0,len(data_test),4):
                result_dict = {}
                prompt = data_test[i]['input']
                sampling_params = SamplingParams(max_tokens=2000,n=5)
                # prompts = [prompt]
                instruction = "Write Python code to solve the question. "
                # instruction = "Write Python code to solve the question.\nThe returned value of the program is supposed to be the answer. It should be integer or float or list of integer/float.\n"

                if args.few_shot:
                    prompts = [instruction + "\n" + math_prompt.MATH_PROMPT_FS + "\nThe question is:\n" + data_test[j]['input']
                                + "\nThe solution code is:\n" for j in range(i,min(i+4, len(data_test)))]
                else:
                    prompts = [instruction + "\nThe question is:\n" + data_test[j]['input']
                                + "\nThe solution code is:\n" for j in range(i,min(i+4, len(data_test)))]
                outputs = llm.generate(prompts, sampling_params)
                # for _ in range(5):
                for j, output in enumerate(outputs):
                    response_list = output.outputs
                    for response in response_list:
                        response_text = response.text
                        # response = llm.generate(prompts, sampling_params)[0].outputs[0].text
                        # generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
                        # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        result_dict = {}
                        # response = response.split(prompt)[1].strip()
                        response_text = response_text.strip()
                        result_dict['id'] = i+j
                        result_dict['question'] = data_test[i+j]['input']
                        result_dict['response'] = response_text
                        result_dict['target'] = data_test[i+j]['label']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)

                        result.append(result_dict)
                        # print("-----")
                        # print(data_test[i]['output'])
                        print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))


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