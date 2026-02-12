import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from vmoe.checkpoints import partitioned

def load_checkpoint(path, tree=None):
    if path.startswith("gs://"):
        print(f"Loading checkpoint: {path}")
        return partitioned.restore_checkpoint(prefix=path, tree=tree)
    elif path.endswith(".pkl"):
        print(f"Loading checkpoint: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported checkpoint format.")

def compute_router_norm_change(c_pre, c_ft, moe_layers, n_experts):

    norm_diff = np.zeros((len(moe_layers), n_experts), dtype=np.float32)

    for idx, layer in enumerate(moe_layers):

        router_1 = c_pre['Encoder'][f'encoderblock_{layer}']['Moe']['Router']['dense']['kernel']

        router_2 = c_ft['Encoder'][f'encoderblock_{layer}']['Moe']['Router']['dense']['kernel']

        diff = (
            jnp.linalg.norm(router_2, ord=2, axis=0)
            - jnp.linalg.norm(router_1, ord=2, axis=0)
        )

        norm_diff[idx] = np.array(diff)

    return norm_diff

def rank_experts(moe_layers, n_experts, values=None, seed=0):

    if values is not None:
        return np.argsort(values, axis=1)

    else:
        print('No metric is provided, ranking experts randomly')
        rng = np.random.default_rng(seed)
        ranks = np.zeros_like(np.zeros((len(moe_layers), n_experts), dtype=np.float32), dtype=int)
        for i in range(ranks.shape[0]):
            ranks[i] = rng.permutation(ranks.shape[1])
        return ranks


def prune_experts(checkpoint, moe_layers, keep_indices, args):
    """
    keep_indices: list[list[int]]
        Experts to KEEP per layer.
    """

    for layer_i, layer in enumerate(moe_layers):

        keep = keep_indices[layer_i]

        routers = keep

        for router in range(args.num_experts_per_layer):
            if router not in keep:
                routers=np.append(routers,router)
        
        block = checkpoint['Encoder'][f'encoderblock_{layer}']['Moe']

        # Router
        block['Router']['dense']['kernel'] = block['Router']['dense']['kernel'][:, routers]

        # MLP Dense_0
        block['Mlp']['Dense_0']['bias'] = block['Mlp']['Dense_0']['bias'][keep, :]

        block['Mlp']['Dense_0']['kernel'] = block['Mlp']['Dense_0']['kernel'][keep, :, :]

        # MLP Dense_1
        block['Mlp']['Dense_1']['bias'] = block['Mlp']['Dense_1']['bias'][keep, :]

        block['Mlp']['Dense_1']['kernel'] = block['Mlp']['Dense_1']['kernel'][keep, :, :]

    return checkpoint


def main(args):

    # MoE layers to prune
    moe_layers = list(map(int, args.moe_layers_to_prune.split(",")))

    # Load checkpoints
    if args.pretrained_ckpt is not None:
        c_pre = load_checkpoint(args.pretrained_ckpt)
    c_ft  = load_checkpoint(args.finetuned_ckpt)

    if args.pruning_method == "router_norm_change":
        if args.pretrained_ckpt is not None:
            norm_diff = compute_router_norm_change(
                c_pre,
                c_ft,
                moe_layers,
                args.num_experts_per_layer
            )
        else:
            raise ValueError("Router norm change based pruning requires a pretrained checkpoint.")
            
        ranks = rank_experts(moe_layers, args.num_experts_per_layer, values=norm_diff)
    else:
        ranks = rank_experts(moe_layers, args.num_experts_per_layer)

    # Number of unpruned experts
    keep_k = args.num_experts_per_layer - args.num_experts_to_prune_per_layer

    keep_indices = [
        ranks[layer_i][-keep_k:]
        for layer_i in range(len(moe_layers))
    ]

    print("Experts kept per layer:")
    for l, k in zip(moe_layers, keep_indices):
        print(f"Layer {l}: {k}")

    # Load finetuned checkpoint for pruning
    pruned_ckpt = load_checkpoint(args.finetuned_ckpt)

    pruned_ckpt = prune_experts(
        pruned_ckpt,
        moe_layers,
        keep_indices,
        args
    )

    # Save
    with open(args.output, "wb") as f:
        pickle.dump(pruned_ckpt, f)

    print(f"\nSaved pruned checkpoint → {args.output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--finetuned_ckpt", required=True)

    parser.add_argument("--pretrained_ckpt", default=None)

    parser.add_argument("--output",
                        default="pruned_checkpoint.pkl")

    parser.add_argument("--num_experts_per_layer",
                        type=int,
                        default=8)

    parser.add_argument("--num_experts_to_prune_per_layer",
                        type=int,
                        default=2)

    parser.add_argument("--pruning_method",
                        choices=[
                            "router_norm_change",
                            "random"
                        ],
                        default="router_norm_change")

    parser.add_argument("--moe_layers_to_prune",
                        type=str,
                        default="1,3,5,7,9,11",
                        help="Comma-separated MoE layer indices"
                       )

    args = parser.parse_args()

    main(args)