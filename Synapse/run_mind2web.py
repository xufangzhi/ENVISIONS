import pickle
import logging
import argparse
import os
from tqdm import tqdm

from synapse.envs.mind2web.env_utils import load_json
from synapse.agents.mind2web import eval_sample

logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    # 252, 177, 912
    parser.add_argument(
        "--benchmark", type=str, choices=["test_task", "test_website", "test_domain"]
    )
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_memory", action="store_true", default=False)
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/mind2web")
    args.log_dir = os.path.join(current_path, "results/mind2web")

    # Evaluate test set
    assert args.benchmark in ["test_task", "test_website", "test_domain"]
    samples = load_json(args.data_dir, args.benchmark)

    # add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    for sample in samples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

    if args.end_idx is None:
        args.end_idx = len(samples)
    for i in tqdm(range(args.start_idx, args.end_idx)):
        eval_sample(i, args, samples[i])


if __name__ == "__main__":
    main()
