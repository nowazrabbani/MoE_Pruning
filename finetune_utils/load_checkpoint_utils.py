from typing import Any, Optional, Union

import jax
import numpy as np
from vmoe.checkpoints import partitioned
import pickle

import multiprocessing.pool

from vmoe.initialization import mapping

from vmoe.checkpoints import serialization
from vmoe import utils

PyTree = Any
ThreadPool = multiprocessing.pool.ThreadPool
Rules = Union[mapping.Rules, mapping.UnparsedRules]
tree_map = jax.tree_util.tree_map
from_state_dict = serialization.from_state_dict
to_state_dict = serialization.to_state_dict
tree_flatten = jax.tree_util.tree_flatten
safe_map = utils.safe_map
safe_zip = utils.safe_zip

def initialize_from_pruned_vmoe(
    *,
    target: PyTree,
    prefix: str,
    rules: Rules,
    thread_pool: Optional[ThreadPool] = None,
    **map_state_dict_kwargs,
) -> PyTree:
  """Initializes the target from a V-MoE checkpoint.

  Args:
    target: PyTree to initialize. This should not be used again once this
      function returns. Use the returned object instead.
    prefix: Filepath of the checkpoint to use for initialization.
    rules: Rules used for mapping variable names from the checkpoint to the
      target.
    mesh: Device mesh used to partition the target PyTree.
    axis_resources_regexes: Optional regexes matching the checkpoint array names
      and specifying how the array read from the checkpoint is partitioned.
      Notice that this is different from the target partitioning, which is
      specified in the target leaves (jax.Array or jax.ShapeDtypeStruct).
    thread_pool: Optional thread pool used to restore checkpoints. This can
      significantly speed-up the time to restore a sharded checkpoint.
    **map_state_dict_kwargs: Additional keyword arguments passed to the
      `mapping.map_state_dict` function.

  Returns:
    A PyTree as the input `target` with (some of) the values loaded from the
    checkpoint.
  """
  ckpt = restore_checkpoint_from_pkl_file(
      prefix=prefix,
      tree=target.params,
      thread_pool=thread_pool)
  return mapping.map_state_dict(ckpt, target, rules, **map_state_dict_kwargs)

def restore_checkpoint_from_pkl_file(
    prefix: str, tree: PyTree,
    *args, **kwargs):
  """Restores from a V1 checkpoint."""
  sharding = tree_map(partitioned._get_array_sharding_or_default, tree)
  sharding_state_dict = to_state_dict(sharding)
  state_dict = _restore_checkpoint_from_pkl(
      prefix, tree, sharding_state_dict, *args, **kwargs)
  return from_state_dict(target=sharding, state=state_dict)

def _restore_checkpoint_from_pkl(
    prefix: str,
    target_tree: PyTree,
    sharding: PyTree,
    thread_pool: Optional[ThreadPool] = None,
) -> PyTree:
  """Restores a PyTree of partitioned arrays from a .pkl file."""
  thread_pool = thread_pool or ThreadPool()
  shapes_dtypes, struct = tree_flatten(target_tree)
  shardings, _ = tree_flatten(sharding)
  # Create Numpy arrays to store the values of each of the addressable array
  # shards addressable.
  local_buffers = [
      partitioned._create_local_buffers(s, i.shape, i.dtype)
      for s, i in safe_zip(shardings, shapes_dtypes)
  ]
  ckpt_pkl = pickle.load(open(prefix+'.pkl','rb'))

  key_lsts = [list(i.keys()) for i in local_buffers]

  ckpt_pkl_flat, _ = tree_flatten(ckpt_pkl)

  for lst in range(len(key_lsts)):
      for slcs in range(len(key_lsts[lst])):
          local_buffers[lst][key_lsts[lst][slcs]] = np.array(ckpt_pkl_flat[lst]
                                                           [((shapes_dtypes[lst].shape[0]//len(key_lsts[lst]))*slcs):
                                                           ((shapes_dtypes[lst].shape[0]//len(key_lsts[lst]))*(slcs+1))],
                                                           dtype = local_buffers[lst][key_lsts[lst][slcs]].dtype)
  del ckpt_pkl_flat, ckpt_pkl
  # Create JAX Arrays from the Numpy buffers.
  def _make_jax_array(i: int) -> jax.Array:
    shape, sharding = shapes_dtypes[i].shape, shardings[i]
    cb = lambda idx: local_buffers[i][partitioned._shard_index_to_slicend(idx, shape)]
    array = jax.make_array_from_callback(shape, sharding, cb)
    local_buffers[i] = None
    return array
  arrays = thread_pool.map(_make_jax_array, range(len(shapes_dtypes)))
  return struct.unflatten(arrays)