# evaluation in MiniWob environment
# Note1: the click position of MiniWoBCoordClick is the offset from body element, which is related to the
# window size of chrome (the window size could be checked in get_screenshot function in env packages).
# Note2: server without Graphical User Interface need to evaluate with the headless mode.
# Note3: if a lot of html code appears and gets stuck, try to disable the proxy.

import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
import numpy as np
from vllm import LLM, SamplingParams

import sys
sys.path.append('..')
from synapse.envs.miniwob.environment import MiniWoBEnv
import synapse.utils.llm
from synapse.agents.miniwob_seeclick import Agent
from synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
    MiniWoBCoordClick,
    MiniWoBElementClickId,
)

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True)
# parser.add_argument('--qwen_path', type=str, required=True)
# parser.add_argument('--imgs_dir_temp', type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--env_name", type=str, default="simple-algebra")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--no_filter", action="store_true", default=True)
parser.add_argument("--no_memory", action="store_true", default=True)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
#
# model_path = args.model_path
# qwen_path = args.qwen_path
# # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval() # load with model checkpoint
# model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()  # load with lora checkpoint
# tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

# PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"

# llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
# sampling_params = SamplingParams(max_tokens=2000,n=1)

# miniwob_imgs_dir_temp = args.imgs_dir_temp
# if not os.path.exists(miniwob_imgs_dir_temp):
#     os.makedirs(miniwob_imgs_dir_temp)
miniwob_train = json.load(open('data/miniwob_data_train.json', 'r'))     # load tasks from train set
miniwob_tasks = list(miniwob_train.keys())
if args.env_name != "all" and args.env_name not in miniwob_tasks:
    miniwob_tasks.append(args.env_name)
task_max_step = {k: (10 if (k != 'guess-number' and k != 'use-slider' and k != 'choose-date') else 30) for k in
                 miniwob_tasks}
# prompt_origin = "Please generate the next move according to the UI html, instruction and previous actions. Instruction: {}. Previous actions: {}"
prompt_origin = "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```"

prompt_origin += f"Next, I will give you two samples for reference."

# few_shot_samples = f"Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n    <div id=\"query\">Click on the \"okay\" button.</div>\n    <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><span data-wob_ref=\"4\" data-wob_eps=\"e0\"> pulvinar neque, lacinia: </span><input type=\"text\" style=\"width:44px;\" data-wob_ref=\"5\" data-wob_eps=\"e0\"><br><button data-wob_ref=\"6\" data-wob_eps=\"e0\">okay</button><br><button data-wob_ref=\"7\" data-wob_eps=\"e0\">Submit</button><div data-wob_ref=\"8\" data-wob_eps=\"e0\"> fermentum sit pulvinar</div><button data-wob_ref=\"9\" data-wob_eps=\"e0\">Previous</button><div data-wob_ref=\"10\" data-wob_eps=\"e0\"> dolor massa tempus</div></div>\n  </div>\nAction:" \
#     + "```\nagent.click_xpath(\"//button[text()='okay']\")\n```" \
#     + "Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n    <div id=\"query\">Click on the \"submit\" button.</div>\n    <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><button data-wob_ref=\"4\" data-wob_eps=\"e0\">submit</button><button data-wob_ref=\"5\" data-wob_eps=\"e0\">previous</button><br><input type=\"text\" style=\"width:83px;\" data-wob_ref=\"6\" data-wob_eps=\"e0\"><br><button data-wob_ref=\"7\" data-wob_eps=\"e0\">Ok</button><div data-wob_ref=\"8\" data-wob_eps=\"e0\"> platea commodo ornare</div><button data-wob_ref=\"9\" data-wob_eps=\"e0\">Yes</button><br></div>\n  </div>\nAction:" \
#     + "```\nagent.click_xpath(\"//button[text()='submit']\")\n```" \
#     + "Here is the test sample. Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n    <div id=\"query\">Click on the \"Ok\" button.</div>\n    <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><input type=\"text\" style=\"width:132px;\" data-wob_ref=\"4\" data-wob_eps=\"e0\"><br><button data-wob_ref=\"5\" data-wob_eps=\"e0\">Ok</button><div data-wob_ref=\"6\" data-wob_eps=\"e0\"> commodo ullamcorper turpis</div><button data-wob_ref=\"7\" data-wob_eps=\"e0\">next</button><br><button data-wob_ref=\"8\" data-wob_eps=\"e0\">submit</button><input type=\"text\" style=\"width:118px;\" data-wob_ref=\"9\" data-wob_eps=\"e0\"><br></div>\n  </div>\nAction:"

few_shot_samples = f"Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n  <div id=\"query\">Select Zsa Zsa from the scroll list and click Submit.</div>\n  <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><select id=\"options\" style=\"width:150px;\" multiple=\"\" data-wob_ref=\"4\" data-wob_eps=\"e0\"><option data-wob_ref=\"5\" data-wob_eps=\"e0\">Bibbie</option><option data-wob_ref=\"6\" data-wob_eps=\"e0\">Hadria</option><option data-wob_ref=\"7\" data-wob_eps=\"e0\">Orsola</option><option data-wob_ref=\"8\" data-wob_eps=\"e0\">Darya</option><option data-wob_ref=\"9\" data-wob_eps=\"e0\">Brandice</option><option data-wob_ref=\"10\" data-wob_eps=\"e0\">Kirstin</option><option data-wob_ref=\"11\" data-wob_eps=\"e0\">Gracie</option><option data-wob_ref=\"12\" data-wob_eps=\"e0\">Zsa Zsa</option><option data-wob_ref=\"13\" data-wob_eps=\"e0\">Jeane</option></select><button class=\"secondary-action\" data-wob_ref=\"14\" data-wob_eps=\"e0\">Submit</button></div>\n</div>\nAction:" \
    + "```\nagent.click_option(\"//option[text() = 'Zsa Zsa']\")\nagent.click_xpath(\"//*[@class='secondary-action']\")\n```\n\n" \
    + "Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n  <div id=\"query\">Select Cornelle, Floria from the scroll list and click Submit.</div>\n  <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><select id=\"options\" style=\"width:150px;\" multiple=\"\" data-wob_ref=\"4\" data-wob_eps=\"e0\"><option data-wob_ref=\"5\" data-wob_eps=\"e0\">Joanie</option><option data-wob_ref=\"6\" data-wob_eps=\"e0\">Sadella</option><option data-wob_ref=\"7\" data-wob_eps=\"e0\">Floria</option><option data-wob_ref=\"8\" data-wob_eps=\"e0\">Lea</option><option data-wob_ref=\"9\" data-wob_eps=\"e0\">Charis</option><option data-wob_ref=\"10\" data-wob_eps=\"e0\">Shalna</option><option data-wob_ref=\"11\" data-wob_eps=\"e0\">Libbey</option><option data-wob_ref=\"12\" data-wob_eps=\"e0\">Cornelle</option><option data-wob_ref=\"13\" data-wob_eps=\"e0\">Gert</option></select><button class=\"secondary-action\" data-wob_ref=\"14\" data-wob_eps=\"e0\">Submit</button></div>\n</div>\nAction:" \
    + "```\nagent.click_option(\"//option[text() = 'Cornelle']\")\nagent.click_option(\"//option[text() = 'Floria']\")\nagent.click_xpath(\"//*[@class='secondary-action']\")\n```\n\n" \
    + "Here is the test sample. Observation:\n<div id=\"wrap\" data-wob_ref=\"2\" data-wob_eps=\"e0\">\n  <div id=\"query\">Select Ferne from the list and click Submit.</div>\n  <div id=\"area\" data-wob_ref=\"3\" data-wob_eps=\"e0\"><select id=\"options\" style=\"width:150px;\" class=\"Valenka,Ferne,Ronda,Kalinda,Sherilyn,Carrissa,Johnath\" data-wob_ref=\"4\" data-wob_eps=\"e0\"><option>Valenka</option><option>Ferne</option><option>Ronda</option><option>Kalinda</option><option>Sherilyn</option><option>Carrissa</option><option>Johnath</option></select><button class=\"secondary-action\" data-wob_ref=\"5\" data-wob_eps=\"e0\">Submit</button></div>\n</div>\nAction:"


# print(prompts)
# response = llm.generate(prompt_origin+few_shot_samples, sampling_params)[0].outputs[0].text
# print(response)

miniwob_tasks = ["simple-algebra"]
result = {}
for env in tqdm(miniwob_tasks):
    if args.env_name != "all":
        if env != args.env_name:
            continue

    success = 0
    print("Task: " + env)
    for j in tqdm(range(args.num_episodes)):
        traj = []
        # initial MiniWob environment
        # seed_task = random.randint(0, 1000000)
        seed_task = 5
        agent = Agent(args=args)
        obs = agent.reset(seed=seed_task)   # return html
        print(obs)
        reward = 0
        previous_actions = []
        for k in range(task_max_step[env]):
            # get the current state
            # miniwob_state = agent.instance.get_state()
            # goal = miniwob_state.utterance
            # print(goal)
            # # agent generate the next action
            # previous_step = ""
            # for i, action in enumerate(previous_actions[-4:]):
            #     previous_step += 'Step' + str(i) + ': ' + action + ". "
            # prompts = prompt_origin.format(goal, previous_step)
            #
            # # query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}, ])
            #
            # print(prompts)
            # response = llm.generate(prompts, sampling_params)[0].outputs[0].text
            # print(response)
            # action_step_record = {"img_path": img_path, "sentence": response, "success": False}
            # traj.append(action_step_record)
            #
            # previous_actions.append(response)
            response = "agent.click_xpath(\"//*[@id='math-answer']\")  # Locate the input box by clicking on it\nagent.type('-1')  # Type the answer in the input box\nagent.click_xpath(\"//*[@id='subbtn']\")\n"


            # try:
            #     action_pred = ast.literal_eval(response)
            # except:
            #     continue
            print(response)
            exec(response)   # 执行actions的同时会在agent.record_traj中记录

            if agent.reward > 0.8:
                print("success")
                print(agent.reward)
                print(agent.record_traj)

            # convert the predicted action to miniwob action that operate the chrome
            try:
                if action_pred["action_type"] == 4:
                    # the offset (150, 105) here is depended on the window size of chrome
                    click_x = action_pred['click_point'][0] * 160
                    click_y = action_pred['click_point'][1] * 210
                    miniwob_action = MiniWoBCoordClick(click_x - 150, click_y - 105)
                elif action_pred["action_type"] == 3:
                    typed_text = action_pred['typed_text']
                    miniwob_action = MiniWoBType(typed_text)
                else:
                    print("action undefined")
                    input()
                    continue
                # execute the action and
                _, reward, done, _ = agent.step(miniwob_action)
            except:
                continue
            # determine if the episode is over, success (reward > 0.8) or fail (done)
            if reward > 0.8:
                success += 1
                for item in traj:
                    item["success"] = True
                break

            if done:
                break

        agent.close()

    result[env] = success / args.num_episodes
    print("Task: " + env + "  Score: " + str(success / args.num_episodes))

print(result)
print("Average Score: " + str(np.mean(list(result.values()))))