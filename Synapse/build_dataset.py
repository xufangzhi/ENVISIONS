import logging
import pickle
import argparse
import os
from datasets import Dataset
from tqdm import tqdm

from synapse.envs.mind2web.env_utils import (
    load_json,
    get_top_k_obs,
    get_target_obs_and_act,
)

logger = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--previous_top_k", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=20)
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["train", "test_task", "test_website", "test_domain"],
    )

    return parser


def build_dataset(args, samples):
    input_dataset = []
    output_dataset = []
    system_prompt = "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id."

    for sample in tqdm(samples):
        prev_actions = []
        prev_obs = []
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            _, target_act = get_target_obs_and_act(s)
            pos_candidates = [
                c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
            ]
            if args.no_trajectory:
                previous_k = 5
                # Continue next loop if the ground truth element is not in the cleaned html
                if len(pos_candidates) != 0:
                    obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
                    query = f"<<SYS>>\n{system_prompt}\n<</SYS>> </s>\n\n"
                    query += f"<s>[INST] Observation:\n```\n{obs}\n```\nTask: {sample['confirmed_task']}\nPrevious actions:\n"
                    if len(prev_actions) > 0:
                        for a in prev_actions[-previous_k:]:
                            query += f"{a}\n"
                    else:
                        query += "None\n"
                    input_dataset.append(query + "Next action: [/INST] ")
                    output_dataset.append("`" + target_act + "` </s>")
                prev_actions.append(act_repr)

            else:
                # Continue next loop if the ground truth element is not in the cleaned html
                if len(pos_candidates) != 0:
                    obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
                    query = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                    for i in range(len(prev_obs)):
                        o, a = prev_obs[i], prev_actions[i]
                        if i == 0:
                            query += (
                                f"[INST]\nTask: {sample['confirmed_task']}\nTrajectory:\n"
                                + o
                                + "\n[/INST]\n"
                            )
                        else:
                            query += "[INST]\n" + o + "\n[/INST]\n"
                        query += a + "\n"
                    if len(prev_obs) == 0:
                        query += (
                            f"[INST]\nTask: {sample['confirmed_task']}\nTrajectory:\nObservation: `"
                            + obs
                            + "`\n[/INST]\n"
                        )
                    else:
                        query += "[INST]\nObservation: `" + obs + "`\n[/INST]\n"
                    input_dataset.append(query)
                    output_dataset.append("Action: `" + target_act + "` </s>")
                target_obs, _ = get_top_k_obs(s, args.previous_top_k)
                prev_obs.append("Observation: `" + target_obs + "`")
                prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

    return Dataset.from_dict({"input": input_dataset, "output": output_dataset})


def main():
    parser = create_parser()
    args = parser.parse_args()
    samples = load_json(args.data_dir, args.benchmark)

    # add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    print("Assigning scores to each candidate")
    for sample in tqdm(samples):
        for s in sample["actions"]:
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]
    print("Building the dataset")
    dataset = build_dataset(args, samples)
    dataset.save_to_disk(
        os.path.join(
            args.data_dir,
            f"{args.benchmark}/{'naive' if args.no_trajectory else 'trajectory'}_top{args.top_k_elements}",
        )
    )
    print("Done")


if __name__ == "__main__":
    main()
