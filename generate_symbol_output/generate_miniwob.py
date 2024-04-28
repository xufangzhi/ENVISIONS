from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
import argparse
import logging
import os
import json
import random
import time
from tqdm import tqdm
import re
import sys
sys.path.append('symbol-llm-v2/Synapse')
from synapse.agents.miniwob_seeclick import Agent


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--env_name", type=str, default="click-collapsible")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--part_id",type=int,default=1,help="part_id to determine the driver",)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_filter", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true", default=True)

    return parser


def extract_seed_from_string(s):
    # Use regular expression to find the number after 'seed'
    match = re.search(r'seed(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_action_blocks(text):
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return text.strip()


# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/3a811845d89f3c1b3f41b341d0f9f05104769f35"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/miniwob_v17_llama2chat13b_sft_iter4_sft_tune_llama2chat_13B"
PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/miniwob_gpt-4-0125-preview_sft_tune_llama2chat_7B"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"
# PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--microsoft--phi-2/snapshots/b10c3eba545ad279e7208ee3a5d644566f001670"

model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
device = "cuda:0"
model.to(device)
batch_size = 1
beam = 1

# unavailable_envs = ["use-autocomplete", "click-menu", "click-pie", ]
unavailable_envs = ['use-autocomplete', "click-menu", "click-pie", 'book-flight', 'login-user-popup', 'login-user', 'terminal', 'click-collapsible-2', \
                    "email-inbox-forward-nl", "social-media-some", "email-inbox-forward-nl-turk", "email-inbox", "email-inbox-nl-turk", \
                    "tic-tac-toe", "click-tab-2", "social-media", "count-shape", "click-tab-2-hard", "find-word", "grid-coordinate", "click-checkboxes-large"]

with open(f"symbol-llm-v2/Synapse/data/miniwob_few_shot_prompt.json",'r') as file:
    few_shot_prompt_dict = json.load(file)


def main():
    parser = create_parser()
    args = parser.parse_args()
    # part = 1
    # logging.basicConfig(filename=f'symbol-llm-v2/Synapse/logs/output_generate_miniwob_llama2chat13b_fs.log', level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     filemode='w')
    logging.basicConfig(filename=f'symbol-llm-v2/Synapse/logs/output_generate_miniwob_gpt-4-0125-preview_sft_tune_llama2chat_7b.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    # model_folder = f"llama2chat_fs_13b_beam{beam}"
    # model_folder = f"phi3_3b_fs_beam{beam}"
    # model_folder = f"miniwob_v17_llama2chat13b_sft_iter4_sft_tune_llama2chat_13b_beam{beam}"
    model_folder = f"miniwob_gpt-4-0125-preview_sft_tune_llama2chat_7b_beam{beam}"
    envs_all = os.listdir('./symbol-llm-v2/Synapse/synapse_rollout')
    envs_filter = [x for x in envs_all if x not in unavailable_envs]
    envs_num = 0
    print(len(envs_filter))

    # if part<=2:
    #     envs_list = envs_filter[15*(part-1):15*part]
    # else:
    #     envs_list = envs_filter[15*(part-1):]

    result = []
    invalid_envs = []
    for envs in tqdm(envs_filter):
        if envs == '.DS_Store' or envs in unavailable_envs:
            continue
        print("="*50)
        print("Env: " + envs)
        logging.info(f'==============')
        logging.info(f'Env: {envs}')
        args.env_name = envs
        agent = Agent(args=args)
        success_rate = 0
        envs_num += 1
        for i in range(args.num_episodes):
            result_dict = {}
            result_dict = {}   # contain reconstructed train sample
            result_dict['env_name'] = envs
            result_dict['seed'] = str(i+args.seed)
            result_dict['status'] = "failure"
            print("Seed: " + str(i+args.seed))

            try:
                obs = agent.reset(seed=int(i+args.seed))
                # print(obs)
            except:
                continue
            if "<div" not in obs:
                invalid_envs.append(envs)
            # output generation
            instruction = "You are required to navigate the web. To accomplish the task, use methods in Agent class to generate actions, with the following functions. type(characters: str): Type a string via the keyboard. click_xpath(xpath: str): Click an HTML element with a valid XPath. press(key_type: str): Press a key on the keyboard (enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v). click_option(xpath: str): Click an option HTML element in a list with a valid XPath. movemouse(xpath: str): Move the mouse cursor on an HTML element with a valid XPath.\n"

            batch_prompts = [instruction + few_shot_prompt_dict[envs] + "\nThe observation is:\n" + obs + "\nThe action is:\n"]
            # batch_prompts = [instruction + "\nThe observation is:\n" + obs + "\nThe action is:\n"]
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
            result_dict['obs'] = obs
            result_dict['response'] = extract_action_blocks(batch_generations[0])

            # print(obs)
            # print(result_dict['response'])
            # print(obs)
            action = result_dict['response']
            # print(action)
            try:
                if "agent." not in action or 'state' in action or not action.startswith("agent."):
                    logging.info(f'failure')
                    result_dict['status'] = "failure"
                else:
                    exec(action)  # 执行actions的同时会在agent.record_traj中记录
                    if agent.reward > 0.8:
                        success_rate += 1
                        logging.info(f'success')
                        print("success")
                        result_dict['status'] = "success"
                    else:
                        result_dict['status'] = "failure"
                        logging.info(f'failure')
                        print("fail")
            except:
                result_dict['status'] = "error"
                logging.info(f'error')
                print("error")
                # continue

            result.append(result_dict)

        agent.close()
        # print(success_rate / len(result))
        logging.info(f'{envs} success rate: {success_rate/args.num_episodes}')

    test_result_folder = f"symbol-llm-v2/test_results/{model_folder}/miniwob/"
    if not os.path.exists(test_result_folder):
        os.system(f"mkdir -p {test_result_folder}")
    with open(f"{test_result_folder}/result.json", 'w') as file:
        json.dump(result, file, indent=4)

    print(f"[info] the result file has been saved.")
    print("==========")

if __name__ == "__main__":
    main()
    # print(111)
