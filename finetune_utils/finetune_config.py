import ml_collections
from vmoe.configs.vmoe_paper import common

# Paths to manually downloaded datasets and to the tensorflow_datasets data dir.
TFDS_MANUAL_DIR = None
TFDS_DATA_DIR = None


def get_config(batch_size,
               num_classes,
               image_size,
               evaluate_evry_steps,
               dataset_name,
               train_steps,
               unpruned_experts_per_encoder,
               pruned_model,
               lr_peak,
               lr_end,
               lr_warmup_steps):
  config = common.get_base_config()
  config.evaluate.every_steps = evaluate_evry_steps      # Evaluate every 100 steps.

  config.dataset = ml_collections.ConfigDict()
  pp_common = f'value_range(-1,1)|onehot({num_classes}, inkey="label", outkey="labels")|keep("image", "labels")'
  # Dataset variation used for training.
  if 'cifar' in dataset_name:
      config.dataset.train = get_data_config(
          name=dataset_name,
          split='train[:98%]',
          process=f'decode|inception_crop({image_size})|flip_lr|{pp_common}',
          shuffle_buffer=50_000,
          batch_size=batch_size,
          cache=None)
      # Dataset variation used for validation.
      config.dataset.val = get_data_config(
          name=dataset_name,
          split='train[98%:]',
          process=f'decode|resize({image_size})|{pp_common}',
          shuffle_buffer=None,
          batch_size=batch_size,
          cache='batched')
      # Dataset variation used for test.
      config.dataset.test = get_data_config(
          name=dataset_name,
          split='test',
          process=f'decode|resize({image_size})|{pp_common}',
          shuffle_buffer=None,
          batch_size=batch_size,
          cache='batched')
  elif dataset_name=='imagenet2012':
      # Dataset variation used for training.
      config.dataset.train = get_data_config(
          name=dataset_name,
          split='train[:99%]',
          process=f'decode_jpeg_and_inception_crop({image_size})|flip_lr|{pp_common}',
          shuffle_buffer=50_000,
          batch_size=batch_size,
          cache=None)
      # Dataset variation used for test.
      config.dataset.test = get_data_config(
          name=dataset_name,
          split='validation',
          process=f'decode|resize({image_size})|{pp_common}',
          shuffle_buffer=None,
          batch_size=batch_size,
          cache='batched')
  else:
      raise ValueError("The datset is not supported")
  # Loss used to train the model.
  config.loss = ml_collections.ConfigDict()
  config.loss.name = 'softmax_xent'
  # Fine-tuning steps.
  config.train_steps = train_steps
  # Description of the upstream model to fine-tune.
  config.description = 'ViT-B/16, E=8, K=2, Every 2, 300 Epochs'
  config.model = get_vmoe_config(config.description,num_classes,image_size)

  unpruned = parse_unpruned_experts(unpruned_experts_per_encoder)
  
  config['model']['encoder']['moe']['no_of_unpruned_experts']=unpruned
  config['model']['name']='VisionTransformerMoePruned'
  # Model initialization from the released checkpoints.
  config.initialization = ml_collections.ConfigDict({
      'name': 'initialize_from_vmoe',
      'prefix': 'gs://vmoe_checkpoints/vmoe_b16_imagenet21k_randaug_strong',
      'rules': [
          ('head', ''),              # Do not restore the head params.
          # We pre-trained on 224px and are finetuning on 384px.
          # Resize positional embeddings.
          ('^(.*/pos_embedding)$', r'params/\1', 'vit_zoom'),
          # Restore the rest of parameters without any transformation.
          ('^(.*)$', r'params/\1'),
      ],
      # We are not initializing several arrays from the new train state, do not
      # raise an exception.
      'raise_if_target_unmatched': False,
      # Partition MoE parameters when reading from the checkpoint.
      'axis_resources_regexes': [('Moe/Mlp/.*', ('expert',))],
  })
  config['initialization']['name'] = 'initialize_from_pruned_vmoe'
  config['initialization']['prefix'] = pruned_model
  config['initialization']['rules'] = config['initialization']['rules'][1:3]
  config.optimizer = ml_collections.ConfigDict({
      'name': 'sgd',
      'momentum': 0.9,
      'accumulator_dtype': 'float32',
      'learning_rate': {
          'schedule': 'warmup_cosine_decay',
          'peak_value': lr_peak,
          'end_value': lr_end,
          'warmup_steps': lr_warmup_steps,
      },
      'gradient_clip': {'global_norm': 10.0},
  })
  # These control how the model parameters are partitioned across the device
  # mesh for running the models efficiently.
  # By setting num_expert_partitions = num_experts, we set at most one expert on
  # each device.
  config.num_expert_partitions = config.model.encoder.moe.num_experts
  # This value specifies that the first axis of all parameters in the MLPs of
  # MoE layers (which has size num_experts) is partitioned across the 'expert'
  # axis of the device mesh.
  config.params_axis_resources = [('Moe/Mlp/.*', ('expert',))]
  config.extra_rng_keys = ('dropout', 'gating')

  return config


def get_data_config(name, split, process, shuffle_buffer, batch_size, cache):
  """Returns dataset parameters."""
  config = common.get_data_config(
      name=name, split=split, process=process, batch_size=batch_size,
      shuffle_buffer=shuffle_buffer, cache=cache)
  config.data_dir = TFDS_DATA_DIR
  config.manual_dir = TFDS_MANUAL_DIR
  return config


def get_vmoe_config(description: str, num_classes: int, image_size: int) -> ml_collections.ConfigDict:
  config = common.get_vmoe_config(description, image_size, num_classes)
  config.representation_size = None
  config.encoder.moe.router.dispatcher.capacity_factor = 1.5
  return config


def get_hyper(hyper):
  return hyper.sweep('config.seed', list(range(3)))

def parse_unpruned_experts(arg):

    if arg is None:
        return None

    result = {}
    pairs = arg.split(",")

    for p in pairs:
        k, v = p.split("=")
        result[k] = int(v)

    return result