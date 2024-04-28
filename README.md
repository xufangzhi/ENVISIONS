<h1 align="center">
<!-- <img src="./logo.png" width="100" alt="Symbol-LLM" /> -->
<br>
Symbol-LLM: Towards Foundational Symbol-centric Interface for Large Language Models
</h1>



<p align="center">
  <a href="https://xufangzhi.github.io/symbol-llm-page/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2311.09278"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/Symbol-LLM/Symbol-LLM-7B-Instruct"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/datasets/Symbol-LLM/Symbolic_Collection"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/xufangzhi/Symbol-LLM"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">Symbol-LLM: Towards Foundational Symbol-centric Interface for Large Language Models</a>"
</p>


## 🔥 News

- [2023/12/28] 🔥🔥🔥 We upload a part of the Symbolic collection (~88K, 10% of the whole collection) on [🤗 HuggingFace](https://huggingface.co/Symbol-LLM/Symbol-LLM-7B-Instruct)! The whole collection is expected to release upon the acceptance of the paper.
- [2023/10/08] 🔥🔥🔥 Model weights of Symbol-LLM are released at [🤗 HuggingFace](https://huggingface.co/datasets/Symbol-LLM/Symbolic_Collection)!
- [2023/11/15] We make the Symbol-LLM paper public !


## 💡 Abstract

<details close>
<summary>Detailed Abstract of Symbol-LLM</summary>

Although Large Language Models (LLMs) demonstrate remarkable ability in processing and generating human-like text, they do have limitations when it comes to comprehending and expressing world knowledge that extends beyond the boundaries of natural language(e.g., chemical molecular formula). Injecting a collection of symbolic data directly into the training of LLMs can be problematic, as it disregards the synergies among different symbolic families and overlooks the need for a balanced mixture of natural and symbolic data. In this work, we tackle these challenges from both a data and framework perspective and introduce Symbol-LLM series models. First, we curated a data collection consisting of 34 tasks and incorporating approximately 20 distinct symbolic families, intending to capture the interrelations and foster synergies between symbols. Then, a two-stage tuning framework succeeds in injecting symbolic knowledge without loss of the generality ability. Extensive experiments on both symbol- and NL-centric tasks demonstrate the balanced and superior performances of Symbol-LLM series models.

</details>

## 🚀 Quick Start

To try on Symbol-LLM, please use the Transformer library:

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Symbol-LLM/Symbol-LLM-7B-Instruct")
```


To utilize our symbolic collection, please load the dataset:

```python
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("Symbol-LLM/Symbolic_Collection")
```

## 📃 Deployed As A WebUI
The implementation of WebUI is modified from [text-generation-webui](https://github.com/oobabooga/text-generation-webui). The running script is as follows:

```bash
cd demo-webui/
python server.py --model <model_name> --api --share --gpu-memory 40 40 --compute_dtype float32 --bf16
```


## 📒 Note
This work is still under review. We will open-source the model weights, symbolic collection and the code.


## 🔧 Repo Structure
This repo contains the training scripts and the demo deployment. Detailed structure is as follow:
```
.
├── README.md
├── logo.png
├── demo-webui
```

## Citation
If you find it helpful, please kindly cite the paper.
```
@article{xu2023symbol,
  title={Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models},
  author={Xu, Fangzhi and Wu, Zhiyong and Sun, Qiushi and Ren, Siyu and Yuan, Fei and Yuan, Shuai and Lin, Qika and Qiao, Yu and Liu, Jun},
  journal={arXiv preprint arXiv:2311.09278},
  year={2023}
}
```
