<h1 align="center">
<!-- <img src="./logo.png" width="100" alt="Symbol-LLM" /> -->
<br>
From Weak to Strong: An Efficient Neural-Symbolic Self-Training Framework
</h1>



<p align="center">
  <a href="https://xufangzhi.github.io/symbol-llm-page/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2311.09278"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/Symbol-LLM/Symbol-LLM-7B-Instruct"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/datasets/Symbol-LLM/Symbolic_Collection"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/xufangzhi/Self-Training"><b>[🐱 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2311.09278" target="_blank">From Weak to Strong: An Efficient Neural-Symbolic Self-Training Framework</a>"
</p>


## 🔥 News

- [2023/12/28] 🔥🔥🔥 We create a new repo !

## 💡 Abstract

<details close>
<summary>Detailed Abstract of Symbol-LLM</summary>

Although Large Language Models (LLMs) demonstrate remarkable ability in processing and generating human-like text, they do have limitations when it comes to comprehending and expressing world knowledge that extends beyond the boundaries of natural language(e.g., chemical molecular formula). Injecting a collection of symbolic data directly into the training of LLMs can be problematic, as it disregards the synergies among different symbolic families and overlooks the need for a balanced mixture of natural and symbolic data. In this work, we tackle these challenges from both a data and framework perspective and introduce Symbol-LLM series models. First, we curated a data collection consisting of 34 tasks and incorporating approximately 20 distinct symbolic families, intending to capture the interrelations and foster synergies between symbols. Then, a two-stage tuning framework succeeds in injecting symbolic knowledge without loss of the generality ability. Extensive experiments on both symbol- and NL-centric tasks demonstrate the balanced and superior performances of Symbol-LLM series models.

</details>
```


## 📒 Note
This work is still in progress.


## Citation
If you find it helpful, please kindly cite our previous paper.
```
@article{xu2023symbol,
  title={Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models},
  author={Xu, Fangzhi and Wu, Zhiyong and Sun, Qiushi and Ren, Siyu and Yuan, Fei and Yuan, Shuai and Lin, Qika and Qiao, Yu and Liu, Jun},
  journal={arXiv preprint arXiv:2311.09278},
  year={2023}
}
```
