# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with gating layers."""
from typing import Any, Callable, Mapping, Optional, Type

import jax
from jax import tree_util
import flax.linen as nn
import flax.core.lift
import jax.numpy as jnp
import vmoe.utils
from vmoe.nn.vit_moe import *

def sparse_moe_pruned_spmd(target: nn.transforms.Target,
                    variable_axes: Mapping[flax.core.lift.CollectionFilter,
                                           flax.core.lift.InOutAxis],
                    split_rngs: Mapping[flax.core.lift.PRNGSequenceFilter,
                                        bool],
                    has_aux: bool = False,
                    methods=None):
  """Lift transformation that wraps a target with a Sparse MoE with pruned experts using SPMD"""

  def wrapper(expert_fn: Callable[..., Any]):

    def transformed(scopes, *inputs):
      outputs = flax.core.lift.vmap(
          expert_fn,
          in_axes=0,
          out_axes=0,
          variable_axes=variable_axes,
          split_rngs=split_rngs)(scopes, *inputs)
      if has_aux:
        outputs, aux = outputs
      return (outputs, aux) if has_aux else outputs

    return transformed

  return nn.transforms.lift_transform(wrapper, target, methods=methods)

class MlpMoePrunedBlock(MlpMoeBlock):
  """Sparse MoE layer of MLPs with pruned experts.

  Attributes:
    no_of_experts_after_pruning: Number of experts in the MoE after pruning.
  """
  no_of_experts_after_pruning: int = 8
    
  @nn.compact
  def __call__(self, inputs):
    assert inputs.ndim == 3, f'Expected ndim = 3, but got shape {inputs.shape}'
    # Reshape inputs from (num_seqs, seq_length, hidden_size) to
    # (num_groups, groups_size, hidden_size).
    inputs_shape = inputs.shape
    inputs = inputs.reshape(-1, self.group_size, inputs.shape[-1])
    dispatcher, metrics = self.create_router()(inputs)
    inputs = tree_util.tree_map(dispatcher.dispatch, inputs)[0:self.no_of_experts_after_pruning]
    # Use the dispatcher to apply a MoE of MlpBlocks.
    mlp_moe_layer = sparse_moe_pruned_spmd(
        MlpBlock,
        has_aux=False,
        variable_axes={'params': 0},
        split_rngs=self.create_split_rngs())(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            deterministic=self.deterministic,
            name='Mlp')
    outputs = mlp_moe_layer(inputs)
    outputs = jnp.concatenate([outputs,jnp.zeros((self.num_experts-self.no_of_experts_after_pruning,outputs.shape[1],outputs.shape[2]))],axis=0)
    outputs = tree_util.tree_map(dispatcher.combine, outputs)
    # Reshape outputs from (num_groups, group_size, output_dim) to
    # (num_seqs, seqs_length, output_dim).
    outputs = outputs.reshape(*inputs_shape[:-1], outputs.shape[-1])
    return outputs, metrics

class PrunedEncoderBlock(EncoderBlock):
  """Encoder block with a Sparse MoE of pruned MLPs."""
  no_of_experts_after_pruning: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    # Attention Block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=self.deterministic,
        dropout_rate=self.attention_dropout_rate,
        normalize_qk=self.attention_qk_norm,
        num_heads=self.num_heads,
        name='SelfAttention')(inputs_q=x, inputs_kv=x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
    x = x + inputs
    # MLP-MoE block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    if self.no_of_experts_after_pruning is not None:
        y = self.mlp_block(dtype=self.dtype, deterministic=self.deterministic,
                           no_of_experts_after_pruning=self.no_of_experts_after_pruning)(y)
    else:
        y = self.mlp_block(dtype=self.dtype, deterministic=self.deterministic)(y)
    if isinstance(y, jnp.ndarray):
      return x + y
    else:
      y, metrics = y
      return x + y, metrics

class EncoderMoePruned(EncoderMoe):
  """Transformer encoder with optional blocks of Sparse MoE of MLPs with Pruned Experts.
  """
  @nn.compact
  def __call__(self, inputs):
    assert inputs.ndim == 3, f'Expected ndim = 3, but got shape {inputs.shape}'
    x = self.add_position_emb(inputs)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)

    dense_mlp_params = dict(mlp_dim=self.mlp_dim,
                            dropout_rate=self.dropout_rate)
    moe_mlp_params = {**dense_mlp_params, **(self.moe or {})}
    moe_mlp_layers = moe_mlp_params.pop('layers', ())
    unpruned_experts_in_each_moe_layer = moe_mlp_params.pop('no_of_unpruned_experts')
    dense_mlp_cls = vmoe.utils.partialclass(
        MlpBlock, **dense_mlp_params, name='Mlp')
    moe_mlp_cls = vmoe.utils.partialclass(
        MlpMoePrunedBlock, **moe_mlp_params, name='Moe')
    encoder_block_cls = vmoe.utils.partialclass(
        PrunedEncoderBlock,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        attention_qk_norm=self.attention_qk_norm,
        deterministic=self.deterministic,
        dtype=self.dtype)

    metrics = {}
    for block in range(self.num_layers):
      if block in moe_mlp_layers:
        no_of_unpruned_experts = unpruned_experts_in_each_moe_layer[f'encoderblock_{block}']
        x, metrics[f'encoderblock_{block}'] = encoder_block_cls(
            name=f'encoderblock_{block}', mlp_block=moe_mlp_cls, 
            no_of_experts_after_pruning = no_of_unpruned_experts)(x)
      else:
        x = encoder_block_cls(
            name=f'encoderblock_{block}', mlp_block=dense_mlp_cls)(x)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    # Sum auxiliary losses from all blocks.
    metrics['auxiliary_loss'] = sum(
        m['auxiliary_loss'] for m in metrics.values())
    return encoded, metrics
        
class VisionTransformerMoePruned(VisionTransformerMoe):
  """Vision Transformer with Sparse MoE layers with Pruned Experts.
  """
  encoder_cls: Type[nn.Module] = EncoderMoePruned
