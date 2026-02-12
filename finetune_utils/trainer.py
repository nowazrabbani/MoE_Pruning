import functools
import multiprocessing.pool
import os
import time
from typing import Any, Dict, Optional, Tuple
from absl import logging
from clu import metric_writers
import flax.linen as nn
import jax
from jax.experimental import pjit
import ml_collections
from vmoe import multihost_utils
from vmoe import utils
from vmoe.data import input_pipeline
from vmoe.data import pjit_utils
import tensorflow as tf

import vmoe.train.trainer as trainer
import vit_moe_pruned
from vmoe.initialization import mapping
from finetune_utils.load_checkpoint_utils import initialize_from_pruned_vmoe
import pickle

def create_flax_model(*, config: Dict[str, Any],
                      deterministic: bool) -> nn.Module:
  if 'name' not in config:
    raise KeyError('The model config must have a "name" field.')
  if isinstance(config, ml_collections.ConfigDict):
    config = config.to_dict()
  model_cls = config.pop('name')
  model_cls, args, kwargs = utils.parse_call(model_cls, vit_moe_pruned)
  return model_cls(*args, **kwargs, **config, deterministic=deterministic)

def initialize_train_state_from_checkpoint(
    *,
    train_state: trainer.TrainState,
    name: str,
    mesh: trainer.Mesh,
    thread_pool: Optional[trainer.ThreadPool] = None,
    **kwargs) -> trainer.TrainState:
    _ = kwargs.pop('axis_resources_regexes','')
    return initialize_from_pruned_vmoe(target=train_state, thread_pool=thread_pool,**kwargs)

trainer.initialize_train_state_from_checkpoint = initialize_train_state_from_checkpoint
trainer.create_flax_model = create_flax_model

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str,
                       mesh: trainer.Mesh, writer: metric_writers.MetricWriter, savefile: str):
  """Trains a model and evaluates it periodically."""
  datasets = input_pipeline.get_datasets(config.dataset)
  if 'train' not in datasets:
    raise KeyError(f'You must have a "train" variant of the dataset. '
                   f'Available variants are {sorted(datasets.keys())!r}')
  train_examples = input_pipeline.get_data_num_examples(config.dataset.train)
  train_batch_size = config.dataset.train.batch_size
  train_steps, train_epochs = trainer.get_train_steps_and_epochs(
      train_steps=config.get('train_steps'),
      train_epochs=config.get('train_epochs'),
      train_batch_size=train_batch_size,
      train_examples=train_examples)
  logging.info(
      'Training for %d steps (%g epochs) over %d examples, with a '
      'batch size of %d', train_steps, train_epochs, train_examples,
      train_batch_size)

  # Get the global shape of the image array.
  datataset_element_shape_dtype = pjit_utils.get_dataset_shape_dtype_struct(
      datasets['train'])

  ckpt_manager = trainer.create_checkpoint_manager(
      workdir=workdir, **config.get('save_checkpoint', {}))
  train_state_initialize_fn = trainer.make_create_train_state_fn(
      model=trainer.create_flax_model(config=config.model, deterministic=False),
      optimizer_config=config.optimizer,
      input_shape_dtypes=(datataset_element_shape_dtype['image'],),
      train_steps=train_steps,
      extra_rng_keys=tuple(config.get('extra_rng_keys', [])),
      seed=config.get('seed', 0))
  train_state, last_seen_index = trainer.restore_or_create_train_state(
      ckpt_manager=ckpt_manager,
      initialize_fn=train_state_initialize_fn,
      axis_resources_regexes=config.params_axis_resources,
      thread_pool=trainer.ThreadPool(),
      initialization_kwargs=config.get('initialization'))
  init_step = int(train_state.step)
  logging.info('Initial step = %d', init_step)
  tr_iter = trainer.get_dataset_iterator(
      dataset=datasets['train'],
      prefetch_size=config.dataset.train.get('prefetch_device', 1),
      mesh=mesh,
      last_seen_index=last_seen_index)
  train_loss_fn, eval_loss_fn, label_pred_fn = trainer.get_loss_fn(**config.loss)
  summarizer = trainer.create_tree_summarizer(config.get('summarize_arrays'))
  train_step_fn = functools.partial(
      trainer.train_step,
      loss_fn=train_loss_fn,
      microsteps=config.get('microsteps'),
      summarizer=summarizer)
  # If mixup options are defined, wrap the train_step_fn with mixup.
  if config.get('mixup', {}):
    mixup_config = config.mixup.to_dict()
    train_step_fn = trainer.wrap_train_step_with_mixup(
        train_step_fn,
        partition_spec=jax.sharding.PartitionSpec(mesh.axis_names,),
        **mixup_config)

  train_step_pjit = pjit.pjit(
      fun=train_step_fn,
      out_shardings=(
          jax.tree_util.tree_map(lambda x: x.sharding, train_state),
          None,
      ),
      donate_argnums=(0, 1, 2),
  )

  # Setup hooks.
  profile_hook = trainer.create_profile_hook(
      workdir=workdir, **config.get('profile', {}))
  progress_hook = trainer.create_progress_hook(
      writer=writer, first_step=init_step + 1, train_steps=train_steps,
      **config.get('report_progress', {}))
  evaluation_hook, config_model_eval = trainer.create_evaluation_hook(
      base_model_config=config.model.copy_and_resolve_references(),
      writer=writer,
      progress_hook=progress_hook,
      datasets={name: ds for name, ds in datasets.items() if name != 'train'},
      loss_fn=eval_loss_fn,
      label_pred_fn=label_pred_fn,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('evaluate', {}))
  fewshot_hook, _ = trainer.create_fewshot_hook(
      base_model_config=config_model_eval,
      writer=writer,
      progress_hook=progress_hook,
      first_step=init_step + 1,
      train_steps=train_steps,
      extra_rng_keys=config.get('extra_rng_keys', []),
      **config.get('fewshot', {}))
  # Run checkpoint hook just before starting the loop. This will save the train
  # Explicitly compile train_step here.
  t0 = time.time()
  train_step_pjit = train_step_pjit.lower(
      train_state,
      datataset_element_shape_dtype['image'],
      datataset_element_shape_dtype['labels']).compile()
  t1 = time.time()
  # Report compilation time, and flops and optimal seconds per step and device.
  writer.write_scalars(init_step + 1, {'train/compile_secs': t1 - t0})
  train_step_flops_per_device, train_step_seconds_per_device = (
      utils.get_flops_and_seconds_per_device(train_step_pjit))
  if train_step_flops_per_device:
    writer.write_scalars(
        init_step + 1,
        {'train/step_flops_per_device': train_step_flops_per_device})
  if train_step_seconds_per_device:
    writer.write_scalars(
        init_step + 1,
        {'train/step_seconds_per_device': train_step_seconds_per_device})
  train_cost_fn = trainer.make_train_cost_fn(train_step_pjit)
  for step, batch in zip(range(init_step + 1, train_steps + 1), tr_iter):
    profile_hook(step)
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_state, metrics = train_step_pjit(train_state, batch['image'],
                                             batch['labels'])
    progress_hook(step, scalar_metrics=(
        train_cost_fn(step) | {f'train/{k}': v for k, v in metrics.items()}
    ))
    evaluation_hook(step, params=train_state.params, **train_cost_fn(step))
    fewshot_hook(step, variables={'params': train_state.params},
                 **train_cost_fn(step))
  ckpt_manager.wait_until_finished()
  multihost_utils.sync_devices('training:completed')
  logging.info('Training completed.')
  with open(savefile, 'wb') as fp:
      pickle.dump(train_state.params, fp)