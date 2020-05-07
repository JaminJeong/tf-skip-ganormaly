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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import os
import time

import numpy as np

# from IPython import display

if not os.path.isdir("output"):
  os.mkdir("output")

BUFFER_SIZE = 400
BATCH_SIZE = 16
EPOCHS = 50

from data_augmentation import normalize

## Input Pipeline

# Training Flags (hyperparameter configuration)
model_name = 'dcgan'
dataset_name = 'mnist'
assert dataset_name in ['mnist', 'fashion_mnist']
learning_rate_D = 1e-4
learning_rate_G = 1e-4

# Load training and eval data from tf.keras
if dataset_name == 'mnist':
  (train_images, train_labels), _ = \
      tf.keras.datasets.mnist.load_data()
else:
  (train_images, train_labels), _ = \
      tf.keras.datasets.fashion_mnist.load_data()

test_label_value = [2, 3]
train_images_list = []
train_labels_list = []
test_images = []
test_labels = []

test_len = train_labels.shape[0] // 1000

for idx, (image, label) in enumerate(zip(train_images, train_labels)):
  if idx < train_labels.shape[0] - test_len:
    train_images_list.append(image)
    train_labels_list.append(label)
  else:
    test_images.append(image)
    test_labels.append(label)

for image, label in zip(train_images, train_labels):
  if not label in test_label_value:
    train_images_list.append(image)
    train_labels_list.append(label)

train_images = np.array(train_images_list)
train_labels = np.array(train_labels_list)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

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

OUTPUT_CHANNELS = 3

from model import Generator, Discriminator, generate_images

generator = Generator()

# * **Generator loss**
#   * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
#   * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
#   * This allows the generated image to become structurally similar to the target image.
#   * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

# The training procedure for the generator is shown below:

LAMBDA = 100


def generator_loss(gen_output, target):
  # mean absolute error
  total_gen_loss = tf.reduce_mean(tf.abs(target - gen_output))

  return total_gen_loss


discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def latent_loss(gen_logit, real_logit):
  total_latent_loss = tf.keras.losses.MeanSquaredError(gen_logit, real_logit)

  return total_latent_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Generate Images
#
# Write a function to plot some images during training.
#
# * We pass images from the test dataset to the generator.
# * The generator will then translate the input image into the output.
# * Last step is to plot the predictions and **voila!**

# Note: The `training=True` is intentional here since
# we want the batch statistics while running the model
# on the test dataset. If we use training=False, we will get
# the accumulated statistics learned from the training dataset
# (which we don't want)


# for index, (example_input, example_target) in enumerate(test_dataset.take(1)):
#   generate_images("./output{}.jpg".format(index), generator, example_input, example_target)
#

# ## Training
# 
# * For each example input generate an output.
# * The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
# * Next, we calculate the generator and the discriminator loss.
# * Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
# * Then log the losses to TensorBoard.

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

    gen_total_loss = generator_loss(gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    latent_loss = latent_loss(disc_generated_output, disc_real_output)

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
      generate_images("./output/epoch_{}.jpg".format(epoch), generator, example_input, example_target)
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

get_ipython().system('ls {}'.format(checkpoint_dir))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


## Generate using test dataset

# Run the trained model on a few examples from the test dataset
for idx, (inp, tar) in enumerate(test_dataset.take(5)):
  generate_images("./output/result_{}.jpg".format(idx), generator, inp, tar)

