<h1 align="center">
<!-- <img src="./logo.png" width="100" alt="Symbol-LLM" /> -->
<br>
Interactive Evolution: An Efficient Neural-Symbolic Self-Training Framework for Large Language Models
</h1>



<p align="center">
  <a href="https://xufangzhi.github.io/symbol-llm-page/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2311.09278"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/Symbol-LLM/Symbol-LLM-7B-Instruct"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/datasets/Symbol-LLM/Symbolic_Collection"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/xufangzhi/Self-Training"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">Interactive Evolution: An Efficient Neural-Symbolic Self-Training Framework for Large Language Models</a>"
</p>


## 🔥 News

- [2023/12/28] 🔥🔥🔥 We create a new repo for the code of ENVISIONS!


## 📒 Note
This work is still in progress.


## 🚀 How to Start Training

To try on ENVISIONS, please use the bash script `run_self_training.sh` or directly use the following command:

For **agentic** task MiniWob, please use:
```bash
python symbol-llm-v2/self_training_miniwob.py --base_model "llama2chat" --model_size "7B" --task_prefix "miniwob_v17_llama2chat" --vllm_batchsize 1
```

For **mathematical** tasks, please use:
```bash
python symbol-llm-v2/self_training.py --base_model "llama2chat" --model_size "7B" --task_prefix "gsm_math_full_v17_llama2chat" --vllm_batchsize 1
```

For **logical reasoning** tasks, please use:
```bash
python symbol-llm-v2/self_training_logic.py --base_model "llama2chat" --model_size "7B" --task_prefix "logic_v17_llama2chat" --vllm_batchsize 1
```

## Citation
If you find it helpful, please kindly cite our paper.
```
@article{xu2023symbol,
  title={Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models},
  author={Xu, Fangzhi and Wu, Zhiyong and Sun, Qiushi and Ren, Siyu and Yuan, Fei and Yuan, Shuai and Lin, Qika and Qiao, Yu and Liu, Jun},
  journal={arXiv preprint arXiv:2311.09278},
  year={2023}
}
```
