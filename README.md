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

## Pruning and Saving Checkpoints

Use the script below to prune experts from a finetuned VMoE checkpoint and save the pruned model.

### Example Command

```bash
python prune_and_save_checkpoint.py \
  --finetuned_ckpt gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong_ft_ilsvrc2012 \
  --pretrained_ckpt gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong \
  --output pruned_ckpts/vmoe_ft_imagenet1k_router_norm_change_encoders_1357911_pruned_2_experts.pkl \
  --num_experts_per_layer 8 \
  --num_experts_to_prune_per_layer 2 \
  --pruning_method router_norm_change \
  --moe_layers_to_prune 1,3,5,7,9,11
```

### Arguments

* `--finetuned_ckpt` : Path to the finetuned checkpoint.
* `--pretrained_ckpt` : Path to the pretrained checkpoint (used as reference for pruning).
* `--output` : File path to save the pruned checkpoint.
* `--num_experts_per_layer` : Total number of experts in each MoE layer.
* `--num_experts_to_prune_per_layer` : Number of experts to prune per layer.
* `--pruning_method` : Criterion used for pruning (e.g., `router_norm_change`).
* `--moe_layers_to_prune` : Comma-separated list of MoE encoder layers to prune.

### Output

The script saves a pruned checkpoint at the location specified by `--output`, which can be used for further finetuning or inference.

## Inference on Pruned VMoE Model

Use the following script to run inference using a pruned VMoE checkpoint.

### Example Command

```bash
python inference_on_pruned_vmoe_model.py \
  --dataset imagenet2012 \
  --split test \
  --num_classes 1000 \
  --checkpoint pruned_ckpts/vmoe_ft_imagenet1k_router_norm_change_encoders_1357911_pruned_2_experts.pkl \
  --capacity_factor 1.5 \
  --batch_size 128 \
  --image_size 384 \
  --patch_size 16 \
  --unpruned_experts encoderblock_1=6,encoderblock_3=6,encoderblock_5=6,encoderblock_7=6,encoderblock_9=6,encoderblock_11=6
```

### Arguments

* `--dataset` : Dataset name (e.g., `imagenet2012`).
* `--split` : Dataset split for evaluation (`train` / `val` / `test`).
* `--num_classes` : Number of output classes.
* `--checkpoint` : Path to the pruned checkpoint.
* `--capacity_factor` : MoE routing capacity factor.
* `--batch_size` : Batch size for inference.
* `--image_size` : Input image resolution.
* `--patch_size` : Patch size used in the model.
* `--unpruned_experts` : Number of remaining experts per pruned MoE layer (layer-wise specification).

### Output

The script reports evaluation metrics (e.g., top-1 / top-5 accuracy) on the specified dataset split using the pruned model checkpoint.

