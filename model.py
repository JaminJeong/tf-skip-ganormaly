import tensorflow as tf
from tensorflow.keras import layers

IMG_WIDTH = 28
IMG_HEIGHT = 28

class Conv(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides, padding='same',
               apply_batchnorm=True, activation='relu'):
    super(Conv, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    assert activation in ['relu', 'leaky_relu', 'none']
    self.activation = activation

    self.conv = layers.Conv2D(filters=filters,
                              kernel_size=(kernel_size, kernel_size),
                              strides=strides,
                              padding=padding,
                              kernel_initializer=tf.random_normal_initializer(0., 0.02),
                              use_bias=not self.apply_batchnorm)
    if self.apply_batchnorm:
      self.batchnorm = layers.BatchNormalization()

  def call(self, x, training=True):
    # convolution
    x = self.conv(x)

    # batchnorm
    if self.apply_batchnorm:
      x = self.batchnorm(x, training=training)

    # activation
    if self.activation == 'relu':
      x = tf.nn.relu(x)
    elif self.activation == 'leaky_relu':
      x = tf.nn.leaky_relu(x)
    else:
      pass

    return x


class ConvTranspose(tf.keras.Model):
  def __init__(self, filters, kernel_size, padding='same',
               apply_batchnorm=True, activation='relu'):
    super(ConvTranspose, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    assert activation in ['relu', 'sigmoid', 'tanh']
    self.activation = activation
    self.up_conv = layers.Conv2DTranspose(filters=filters,
                                          kernel_size=(kernel_size, kernel_size),
                                          strides=2,
                                          padding=padding,
                                          kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                          use_bias=not self.apply_batchnorm)
    if self.apply_batchnorm:
      self.batchnorm = layers.BatchNormalization()

  def call(self, x, training=True):
    # conv transpose
    x = self.up_conv(x)

    # batchnorm
    if self.apply_batchnorm:
      x = self.batchnorm(x, training=training)

    # activation
    if self.activation == 'relu':
      x = tf.nn.relu(x)
    elif self.activation == 'sigmoid':
      x = tf.nn.sigmoid(x)
    else:
      x = tf.nn.tanh(x)

    return x


class Generator(tf.keras.Model):
  """Build a generator that maps latent space to real space.
    G(z): z -> x
  """

  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = Conv(64, 4, 2, apply_batchnorm=False, activation='leaky_relu')
    self.conv2 = Conv(128, 4, 2, activation='leaky_relu')
    self.conv3 = Conv(256, 3, 2, padding='valid', activation='leaky_relu')
    self.conv4 = Conv(512, 3, 1, padding='valid', apply_batchnorm=False)
    self.convTranspose1 = ConvTranspose(256, 3, padding='valid')
    self.convTranspose2 = ConvTranspose(128, 3, padding='valid')
    self.convTranspose3 = ConvTranspose(64, 4)
    self.convTranspose4 = ConvTranspose(1, 4, apply_batchnorm=False, activation='tanh')

  def call(self, inputs, training=True):
    """Run the model."""
    # inputs: [28, 28, 1]
    conv1 = self.conv1(inputs, training=training) # conv1: [14, 14, 64]
    conv2 = self.conv2(conv1, training=training)  # conv2: [7, 7, 128]
    conv3 = self.conv3(conv2, training=training)  # conv3: [3, 3, 256]
    conv4 = self.conv4(conv3, training=training)  # conv3: [1, 1, 512]
    conv_traspose_1 = self.convTranspose1(conv4, training=training)  # conv1: [3, 3, 256]
    conv_traspose_2 = self.convTranspose2(conv_traspose_1, training=training)  # conv2: [7, 7, 128]
    conv_traspose_3 = self.convTranspose3(conv_traspose_2, training=training)  # conv3: [14, 14, 64]
    generated_images = self.convTranspose4(conv_traspose_3, training=training)  # generated_images: [28, 28, 1]

    return generated_images


class Discriminator(tf.keras.Model):
  """Build a discriminator that discriminate real image x whether real or fake.
    D(x): x -> [0, 1]
  """

  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = Conv(64, 4, 2, apply_batchnorm=False, activation='leaky_relu')
    self.conv2 = Conv(128, 4, 2, activation='leaky_relu')
    self.conv3 = Conv(256, 3, 2, padding='valid', activation='leaky_relu')
    self.conv4 = Conv(1, 3, 1, padding='valid', apply_batchnorm=False, activation='none')

  def call(self, inputs, training=True):
    """Run the model."""
    # inputs: [28, 28, 1]
    inputs = tf.keras.layers.concatenate(inputs)  # (bs, 256, 256, channels*2)
    conv1 = self.conv1(inputs)  # conv1: [14, 14, 64]
    conv2 = self.conv2(conv1)  # conv2: [7, 7, 128]
    conv3 = self.conv3(conv2)  # conv3: [3, 3, 256]
    conv4 = self.conv4(conv3)  # conv4: [1, 1, 1]
    discriminator_logits = tf.squeeze(conv4, axis=[1, 2])  # discriminator_logits: [1,]

    return discriminator_logits

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def latent_loss(gen_logit, real_logit):
  # total_latent_loss = tf.keras.losses.MeanSquaredError(gen_logit, real_logit)
  total_latent_loss = tf.reduce_mean(tf.square(gen_logit - real_logit))

  return total_latent_loss

def generator_loss(gen_output, target, disc_real_output, disc_generated_output):
  # mean absolute error
  con_loss = tf.reduce_mean(tf.abs(target - gen_output))
  lat_loss = latent_loss(disc_real_output, disc_generated_output)
  adv_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  total_gen_loss = 50 * con_loss + lat_loss + adv_loss

  return total_gen_loss, con_loss, lat_loss, adv_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  lat_loss = latent_loss(disc_real_output, disc_generated_output)
  total_disc_loss = real_loss + generated_loss + lat_loss

  return total_disc_loss, real_loss, generated_loss, lat_loss


def generate_images(save_path, model, test_input, tar=None):
  import numpy as np
  import cv2, os
  filepath, ext = os.path.splitext(save_path)
  prediction = model(test_input, training=False)
  len = test_input.shape[0]
  for idx in range(len):
    save_idx_path = filepath + "_" + str(idx) + ext
    result = test_input[idx]
    if not tar is None:
      result = np.concatenate((result, tar[idx]), axis=1)
    result = np.concatenate((result, prediction[idx]), axis=1)
    result = result * 0.5 + 0.5
    result = result * 255
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_idx_path, result)
