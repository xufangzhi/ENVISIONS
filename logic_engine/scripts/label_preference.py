import os
os.chdir("symbol-llm-v2/logic_engine/")

import sys
sys.path.append(os.getcwd())

import json
import re

import numpy as np
import time
import signal
from models.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from models.symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from models.symbolic_solvers.csp_solver.csp_solver import CSP_Program
from models.symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random

import argparse
import logging

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
        default="logic_v17_deepseekchat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--repaired",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot prompting",
    )
    args = parser.parse_args()
    return args


class LogicInferenceEngine:
    def __init__(self):
        self.dataset_name = "ProofWriter"
        program_executor_map = {'FOLIO': FOL_Prover9_Program,
                                'ProntoQA': Pyke_Program,
                                'ProofWriter': Pyke_Program,
                                'LogicalDeduction': CSP_Program,
                                'AR-LSAT': LSAT_Z3_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        # self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy,
        #                                                 self.args.backup_LLM_result_path)

    def safe_execute_program(self, logic_program):
        program = self.program_executor(logic_program, self.dataset_name)
        # cannot parse the program
        if program.flag == False:
            # answer = self.backup_generator.get_backup_answer(id)
            return None, 'failure'
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            # answer = self.backup_generator.get_backup_answer(id)
            return answer, 'failure'
        # successfully executed
        # print(answer)
        # answer = program.answer_mapping(answer)
        return answer, 'success'

    def cleanup(self):
        complied_krb_dir = 'compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')


def extract_ans(text):
    # 根据"Query:"查找文本中的位置
    query_index = text.find("Query:")
    # 如果找到了"Query:"
    if query_index != -1:
        # 在"Query:"之后的文本中查找"\n\n\n"的位置
        position = text[query_index:].find("\n\n\n")

        # 如果找到了"\n\n\n"
        if position != -1:
            # 返回索引之前的内容
            return text[:query_index + position].strip().replace("\n\n\n","\n").replace("\n\n","\n").strip()
    # 如果无法提取内容，返回空字符串
    return text.strip().replace("\n\n\n","\n").replace("\n\n","\n").strip()

def main():
    args = parse_args()

    if args.repaired:
        suffix = "_repaired"
    else:
        suffix = ""

    num = 0
    for part_index in range(1,9):
        part = f"part{part_index}"

        if args.few_shot:
            with open(f"../score_memory/{args.task_prefix}/{args.task_prefix}_{part}_iter0.json") as file:
                pred = json.load(file)
        else:
            with open(f"../../new_generated_data/{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.json") as file:
                pred = json.load(file)

        with open(f"../open-instruct/data/proofwriter_{part}.json") as file:
            gold = json.load(file)

        print(len(pred),len(gold))
        assert len(pred)//5==len(gold), "The length between pred and gold is not equal"

        preference_scores_list = []
        # for each sample
        for i in range(len(gold)):
            preference_scores_each_sample = []
            for j in range(5):
                logic_program = extract_ans(pred[i*5+j]['response'])
                # ground_truth = pred[i*5+j]['label']
                ground_truth = gold[i]['answer']
                # print(logic_program)
                # print(logic_program)
                # print("------")
                # print(i,j)
                engine = LogicInferenceEngine()
                answer, status = engine.safe_execute_program(logic_program)
                # print(answer)
                # engine.cleanup()
                # print(logic_program)
                if answer==ground_truth:
                    # print(answer)
                    num += 1
                    preference_scores_each_sample.append(1)
                else:
                    preference_scores_each_sample.append(0)
            preference_scores_list.append(preference_scores_each_sample)


        if args.few_shot:
            np.save(f"../../symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part}_iter0.npy", np.array(preference_scores_list))
        else:
            np.save(f"../../symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part}_iter{args.cur_iter+1}{suffix}.npy",np.array(preference_scores_list))

    os.chdir("../..")
    print(num)


if __name__ == "__main__":
    main()
    # logic_program = "Predicates:\nGreen($x, bool) ::: Is x green?\nNeeds($x, $y, bool) ::: Does x need y?\nSees($x, $y, bool) ::: Does x see y?\nRed($x, bool) ::: Is x red?\nLikes($x, $y, bool) ::: Does x like y?\nNice($x, bool) ::: Is x nice?\nCold($x, bool) ::: Is x cold?\n\nFacts:\nGreen(Bear, True) ::: The bear is green.\nNeeds(Bear, Cow, True) ::: The bear needs the cow.\nSees(Bear, Cat, True) ::: The bear sees the cat.\nSees(Bear, Mouse, True) ::: The bear sees the mouse.\nRed(Cat, True) ::: The cat is red.\nSees(Cat, Cow, True) ::: The cat sees the cow.\nGreen(Cow, True) ::: The cow is green.\nLikes(Cow, Bear, True) ::: The cow likes the bear.\nNeeds(Cow, Cat, True) ::: The cow needs the cat.\nSees(Cow, Bear, True) ::: The cow sees the bear.\nSees(Cow, Mouse, True) ::: The cow sees the mouse.\nNice(Mouse, True) ::: The mouse is nice.\nLikes(Mouse, Cat, True) ::: The mouse likes the cat.\nNeeds(Mouse, Cow, True) ::: The mouse needs the cow.\n\nRules:\nLikes($x, Cat, True) && Needs($x, Cow, True) >>> Nice($x, True) ::: If someone likes the cat and they need the cow then the cow is nice.\nLikes($x, Mouse, True) >>> Green(Mouse, True) ::: If someone likes the mouse then the mouse is green.\nCold($x, True) >>> Likes($x, Mouse, True) ::: If someone is cold then they like the mouse.\nRed($x, True) >>> Cold($x, True) ::: All red people are cold.\nSees($x, Cat, True) >>> Needs(Cat, Mouse, True) ::: If someone sees the cat then the cat needs the mouse.\nGreen($x, True) >>> Cold($x, True) ::: All green people are cold.\nRed($x, True) >>> Nice($x, True) ::: If someone is red then they are nice.\nLikes($x, Mouse, True) && Sees($x, Cow, True) >>> Cold($x, True) ::: If someone likes the mouse and they see the cow then they are cold.\n\nQuery:\nNice(Cow, False) ::: The cow is not nice."
    #
    # engine = LogicInferenceEngine()
    # answer, status = engine.safe_execute_program(logic_program)
    # print(answer, status)