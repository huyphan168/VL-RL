# *SFT Memorizes, RL Generalizes*: 

## A Comparative Study of Foundation Model Post-training

<div align="center">

<p>
    <img src="assets/teaser.png" alt="Cambrian" width="500" height="auto">
</p>



<a href="https://arxiv.org/abs/2406.16860" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-SFT vs RL-red?logo=arxiv" height="25" />
</a>
<a href="https://tianzhechu.com/SFTvsRL/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-tianzhechu.com/SFTvsRL-blue.svg" height="25" />
</a>
<a href="https://huggingface.co/collections/tianzhechu/sftvsrl-models-and-data-6797ba6de522c7de7fcb80bah" target="_blank">
    <img alt="HF Model: Cambrian-1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Checkpoints&Data-ffc107?color=ffc107&logoColor=white" height="25" />
</a>


<div style="font-family: charter; text-align: center; margin: 0 auto;">
                    <a href="https://tianzhechu.com/" class="author-link" target="_blank">Tianzhe Chu*</a> &emsp;
                    <a href="https://yx-s-z.github.io/" class="author-link" target="_blank">Yuexiang Zhai*</a> &emsp;
                    <a href="https://jihanyang.github.io/" class="author-link" target="_blank">Jihan Yang</a> &emsp;
                    <a href="https://tsb0601.github.io/petertongsb/" class="author-link" target="_blank">Shengbang Tong</a> &emsp;
                    <br>
                    <a href="https://www.sainingxie.com/" class="author-link" target="_blank">Saining Xie</a> &emsp;
                    <a href="https://webdocs.cs.ualberta.ca/~dale/" class="author-link" target="_blank">Dale Schuurmans</a> &emsp;
                    <a href="https://cs.stanford.edu/~quocle/" class="author-link" target="_blank">Quoc V. Le</a> &emsp;
                    <a href="https://people.eecs.berkeley.edu/~svlevine/" class="author-link" target="_blank">Sergey Levine</a> &emsp;
                    <a href="https://people.eecs.berkeley.edu/~yima/" class="author-link" target="_blank">Yi Ma</a> &emsp;
</div>
<br>
</div>


> *Misc: We prompt DALLE-3 via "Conceptual figure of 'SFT Memorizes, RL Generalizes', with trendlines and style of Hong Kong", though Hong Kong-style skyscrapers dominates.*

## Release
- [01/28/25] Excited to shout out our paper *SFT Memorizes, RL Generalizes*! We release the environments, training scripts, evaluation scripts, SFT data, and initial checkpoints.

## Contents
- [Installation](#installation)
- [Train](#train)
- [Evaluation](#evaluation)

## Installation

### Prepare
Our codebase is tested on H800 servers with <code>python 3.13.0</code> <code>torch 2.5.1+cu124</code>.

1. Clone this repository and navigate to into the codebase
```Shell
git clone https://github.com/LeslieTrue/SFTvsRL.git
cd SFTvsRL
```

2. Install Packages
```Shell
conda create -n SFTvsRL python==3.13 -y
conda activate SFTvsRL
pip install -r requirements.txt
cd gym
pip install -e . # install gym environment
cd ..
```

### Download Initial Checkpoints (Optional)
We instantiate RL experiments on top of SFT initialized checkpoints to guarantee model's basic instruction following capabilities. We provide all 4 initial checkpoints for \{GeneralPoints, V-IRL\}$\times$\{Language (-L), Vision-Language (-VL)\}. 
```Shell
huggingface-cli download tianzhechu/GP-L-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/GP-VL-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/VIRL-VL-Init --local-dir YOUR_LOCAL_DIR
huggingface-cli download tianzhechu/GP-VL-Init --local-dir YOUR_LOCAL_DIR
```
It's optional to download these checkpoints via huggingface CLI. You may directly specify <code>repo_name</code> as <code>CKPT_NAME</code> in shell scripts.

## Getting Started

## Citation

If you find Cambrian useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{chu2025sft,
      title={SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training}, 
      author={Tianzhe Chu, },
      year={2025},
      eprint={TBD},
}
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): We start from codebase from the amazing LLaVA
- [Vicuna](https://github.com/lm-sys/FastChat): We thank Vicuna for the initial codebase in LLM and the open-source LLM checkpoitns
- [LLaMA](https://github.com/meta-llama/llama3): We thank LLaMA for continuing contribution to the open-source community and providing LLaMA-3 checkpoints.
- [Yi](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B): We thank Yi for opensourcing very powerful 34B model. 



## Related Projects
- [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://tsb0601.github.io/mmvp_blog/)
- [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://vstar-seal.github.io/)
- [V-IRL: Grounding Virtual Intelligence in Real Life](https://virl-platform.github.io/)



## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/cambrian-mllm/cambrian/blob/main/LICENSE)<br>
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-3, and Vicuna-1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.