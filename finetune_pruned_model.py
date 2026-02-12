import functools

from absl import app
from absl import flags
from finetune_utils import trainer, finetune_config

from absl import logging
from clu import metric_writers
from clu import platform
import jax
import tensorflow as tf
from vmoe import partitioning

flags.DEFINE_string('workdir', None, 'Directory to store logs and model data.')
flags.DEFINE_string("pruned_model", None, "Path to the initial pruned model to be finetuned.")
flags.DEFINE_string(
    'savefile',
    None,
    'File to store finetuned model in PKL format.'
)
flags.DEFINE_string("dataset_name", "cifar10", "Dataset to fine-tune on. Options: 'cifar10', 'cifar100', 'imagenet2012' ")
flags.DEFINE_integer("batch_size", 512, "Batch size for training.")
flags.DEFINE_integer("num_classes", 10, "Number of classes.")
flags.DEFINE_integer("image_size", 128, "Input image size.")
flags.DEFINE_integer("evaluate_evry_steps", 100, "Number of every steps after which the model will be evaluated.")
flags.DEFINE_integer("train_steps", 1000, "Number of total training steps.")
flags.DEFINE_string("unpruned_experts_per_encoder", 
                    "encoderblock_1=2,encoderblock_3=2,encoderblock_5=2,encoderblock_7=2,encoderblock_9=2,encoderblock_11=2", 
                    "Number of experts unpruned in each encoder.")
flags.DEFINE_float(
    "lr_peak", 0.0015, "Peak learning rate for warmup_cosine_decay schedule."
)
flags.DEFINE_float(
    "lr_end", 1e-5, "Final learning rate at the end of training."
)
flags.DEFINE_integer(
    "lr_warmup_steps", 100, "Number of warmup steps for the learning rate schedule."
)
flags.mark_flags_as_required(['workdir', 'pruned_model', 'savefile'])
FLAGS = flags.FLAGS

def run(main):
  jax.config.config_with_absl()
  app.run(functools.partial(_main, main=main))

def _main(argv, *, main) -> None:
  """Runs the `main` method after some initial setup."""
  del argv
  config = finetune_config.get_config(FLAGS.batch_size,
               FLAGS.num_classes,
               FLAGS.image_size,
               FLAGS.evaluate_evry_steps,
               FLAGS.dataset_name,
               FLAGS.train_steps,
               FLAGS.unpruned_experts_per_encoder,
               FLAGS.pruned_model,
               FLAGS.lr_peak,
               FLAGS.lr_end,
               FLAGS.lr_warmup_steps)
  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.set_visible_devices([], 'GPU')
  # Log JAX compilation steps.
  jax.config.update('jax_log_compiles', True)
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  # Log useful information to identify the process running in the logs.
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                     FLAGS.jax_xla_backend)
  logging.info('Using JAX XLA backend %s', jax_xla_backend)
  # Log the configuration passed to the main script.
  logging.info('Config: %s', config)
  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')
  # CLU metric writer.
  logdir = FLAGS.workdir
  writer = metric_writers.create_default_writer(
      logdir=logdir, just_logging=jax.process_index() > 0)
  # Set logical device mesh globally.
  mesh = partitioning.get_auto_logical_mesh(config.num_expert_partitions,
                                            jax.devices())
  partitioning.log_logical_mesh(mesh)
  with metric_writers.ensure_flushes(writer):
    with mesh:
      main(config, FLAGS.workdir, mesh, writer, FLAGS.savefile)

if __name__ == '__main__':
  run(trainer.train_and_evaluate)
