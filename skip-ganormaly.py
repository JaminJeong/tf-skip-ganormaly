#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import os
import time

import argparse

parser = argparse.ArgumentParser(prog="training parameter",
                                 description="training parameter", add_help=True)
parser.add_argument('-t', '--DATATYPE', help='data type.', default='mnist', required=False)
args = parser.parse_args()

from dataset import MnistDataset, FashinMnishDataset

# from IPython import display

if not os.path.isdir("output"):
  os.mkdir("output")

BUFFER_SIZE = 400
BATCH_SIZE = 64
EPOCHS = 50

from data_augmentation import normalize

## Input Pipeline

# Training Flags (hyperparameter configuration)
dataset_name = args.DATATYPE
assert dataset_name in ['mnist', 'fashion_mnist']
learning_rate_D = 1e-4
learning_rate_G = 1e-4

if dataset_name == 'mnist':
    mnist_dataset = MnistDataset()
    train_images, train_labels = mnist_dataset.get_train_data()
    test_images, test_labels = mnist_dataset.get_test_data()
if dataset_name == 'fashion_mnist':
    fashin_mnish_dataset = FashinMnishDataset()
    train_images, train_labels = fashin_mnish_dataset.get_train_data()
    test_images, test_labels = fashin_mnish_dataset.get_test_data()

train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

#train_images = train_images / 255. # Normalize the images to [0, 1]
# train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

def image_augmentation(image):
  input_image, output_image = normalize(image, image)
  return input_image, output_image

#tf.random.set_seed(219)
# for train
N = len(train_images)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.map(image_augmentation,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N)
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
print(train_dataset)

test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
test_dataset = test_dataset.map(image_augmentation,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

from model import Generator, Discriminator, generate_images
from model import generator_loss, discriminator_loss

generator = Generator()
LAMBDA = 100

discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, con_loss, lat_loss, adv_loss = generator_loss(gen_output, target, disc_real_output, disc_generated_output)
    disc_loss, real_loss, generated_loss, _ = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('con_loss', con_loss, step=epoch)
    tf.summary.scalar('lat_loss', lat_loss, step=epoch)
    tf.summary.scalar('adv_loss', adv_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

# The actual training loop:
# 
# * Iterates over the number of epochs.
# * On each epoch it clears the display, and runs `generate_images` to show it's progress.
# * On each epoch it iterates over the training dataset, printing a '.' for each example.
# * It saves a checkpoint every 20 epochs.

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    for example_input, example_target in test_ds.take(1):
      generate_images(f"./output/epoch_{epoch}.jpg", generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


#docs_infra: no_execute
#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('tensorboard', '--logdir {log_dir}')


# Now run the training loop:

fit(train_dataset, EPOCHS, test_dataset)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
