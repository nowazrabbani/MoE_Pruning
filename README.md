# Code for the paper titled "A Provably Effective Method for Pruning Experts in Fine-tuned Sparse Mixture-of-Experts" [ICML'2024]
This repository implements expert-pruning, finetuning, and inference of [Google's Vision Mixture-of-Experts (VMoE) model](https://arxiv.org/pdf/2106.05974) on benchmark vision tasks.

## Installation

Follow the steps below to set up the environment and install dependencies:

```bash
conda create --name moe_pruning
conda activate moe_pruning

git clone https://github.com/nowazrabbani/MoE_Pruning.git
cd moe_pruning

git clone https://github.com/google-research/vmoe.git vision_moe
git clone https://github.com/google-research/vision_transformer.git

mv vision_moe/vmoe .
mv vision_transformer/vit_jax .

cd vision_moe
pip install -r requirements.txt
cd ..

cd vit_jax
pip install -r requirements.txt
cd ..

pip install -q 'jax[cuda]' -f https://storage.googleapis.com/jax-releases/jax_releases.html

rm -rf vision_moe
rm -rf vision_transformer
```

