import os
import tensorflow as tf
import random
import numpy as np

from model import Generator, Discriminator, generate_images
from model import generator_loss

from dataset import MnistDataset, FashinMnistDataset

if __name__ == "__main__":

  print("make generate_image!!")
  if not os.path.isdir("./generate_image"):
    os.mkdir("./generate_image")

  checkpoint_dir = './training_checkpoints'
  print("load models!!")
  generator = Generator()
  discriminator = Discriminator()
  print("load ckpt!!")
  checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  dataset_name = 'fashion_mnist'
  assert dataset_name in ['mnist', 'fashion_mnist']

  if dataset_name == 'mnist':
    mnist_dataset = MnistDataset()
    test_images, test_labels = mnist_dataset.get_test_data()
  if dataset_name == 'fashion_mnist':
    fashin_mnist_dataset = FashinMnistDataset()
    test_images, test_labels = fashin_mnist_dataset.get_test_data()
    random_noise_test_images, _ = fashin_mnist_dataset.get_random_noise_test_data()
    random_noise_test_images = random_noise_test_images.reshape(-1, 28, 28, 1).astype('float32')

  test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

  print("Compute anomaly scores!!")

  for idx, (test_image, test_label) in enumerate(zip(random_noise_test_images, test_labels)):
    if idx % 2 == 0:
      test_image = (test_image / 127.5) - 1
    else:
      test_image = (test_images[idx] / 127.5) - 1
    test_image = np.expand_dims(test_image, axis=0)
    gen_output = generator(test_image, training=False)
    disc_real_output = discriminator([test_image, test_image], training=False)
    disc_generated_output = discriminator([test_image, gen_output], training=False)

    anomaly_score, con_loss, lat_loss, adv_loss = generator_loss(gen_output,
                   test_image,
                   disc_real_output,
                   disc_generated_output)
    generate_images("./generate_image/{}_idx_{}_anomaly_score_{}.jpg".format(test_label, idx, anomaly_score), generator, test_image, test_image)
