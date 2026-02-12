import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from vmoe.data import input_pipeline
from vmoe.configs.vmoe_paper import common

import vit_moe_pruned

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_classes", type=int, required=True)

    # Model / checkpoint
    parser.add_argument("--checkpoint", required=True)

    # MoE capacity factor
    parser.add_argument("--capacity_factor", type=float, default=1.5)

    # Image / batching
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=16)

    # MoE pruning config
    parser.add_argument(
        "--unpruned_experts",
        type=str,
        default=None,
        help=(
            'Example: '
            '"encoderblock_1=2,encoderblock_3=2"'
        )
    )

    return parser.parse_args()


def parse_unpruned_experts(arg):

    if arg is None:
        return None

    result = {}
    pairs = arg.split(",")

    for p in pairs:
        k, v = p.split("=")
        result[k] = int(v)

    return result


def build_model(args):

    description = "ViT-B/16, E=8, K=2, Every 2, 300 Epochs"

    model_config = common.get_vmoe_config(description, args.image_size, args.num_classes)
    model_config.representation_size = None
    model_config.encoder.moe.router.dispatcher.capacity_factor = args.capacity_factor
    model_config = dict(model_config)

    # Insert pruning config
    unpruned = parse_unpruned_experts(
        args.unpruned_experts
    )

    if unpruned is not None:
        model_config["encoder"]["moe"][
            "no_of_unpruned_experts"
        ] = unpruned

    model_config["name"] = "VisionTransformerMoePruned"

    model_cls = getattr(
        vit_moe_pruned,
        model_config.pop("name")
    )

    model = model_cls(
        deterministic=True,
        **model_config
    )

    return model


def get_dataset(args):
    
    if "cifar" in args.dataset:
        process = (
            f'keep("image","label")'
            f'|decode'
            f'|resize({args.image_size}, inkey="image")'
            f'|value_range(-1,1)'
        )
    
        dataset = input_pipeline.get_dataset(
            variant="test",
            name=args.dataset,
            split=args.split,
            batch_size=args.batch_size,
            process=process,
        )
    elif args.dataset=="imagenet2012":
        pp_common = f'value_range(-1,1)|onehot({args.num_classes}, inkey="label", outkey="labels")|keep("image", "labels")'
        data_config = common.get_data_config(name=args.dataset,
                                             split='validation', 
                                             process=f'decode|resize({args.image_size})|{pp_common}',
                                             batch_size=args.batch_size,
                                             shuffle_buffer=None,
                                             cache='batched')
        data_config.data_dir = None
        data_config.manual_dir = None
        del data_config["prefetch_device"]
        data_config['batch_size']=args.batch_size
        dataset = input_pipeline.get_dataset(variant='test',**data_config)
    else:
        raise ValueError("The datset is not supported")

    return dataset


def run_inference(model, checkpoint, dataset, args):

    ncorrect = 0
    ntotal = 0

    for batch in tqdm(dataset):

        mask = batch["__valid__"]

        logits, _ = model.apply(
            {"params": checkpoint},
            batch["image"]
        )

        log_p = jax.nn.softmax(logits)

        preds = jnp.argmax(log_p, axis=1)
        
        if "cifar" in args.dataset:
            ncorrect += jnp.sum(
                (preds == batch["label"]) * mask
            )
        elif args.dataset=="imagenet2012":
            ncorrect += jnp.sum((preds == jnp.argmax(batch['labels'],axis=1)) * mask)
        else:
            raise ValueError("The datset is not supported")

        ntotal += jnp.sum(mask)

    acc = (ncorrect / ntotal) * 100

    return float(acc)


def main():

    args = get_args()

    print("Loading checkpoint…")
    checkpoint = pickle.load(
        open(args.checkpoint, "rb")
    )

    print("Building model…")
    model = build_model(args)

    print("Loading dataset…")
    dataset = get_dataset(args)

    print("Running inference…")
    acc = run_inference(
        model,
        checkpoint,
        dataset,
        args
    )

    print(f"\nTest accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
