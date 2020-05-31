import os
import tensorflow as tf
import random
import numpy as np

from model import Generator, Discriminator, generate_images
from model import generator_loss

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

  dataset_name = 'mnist'
  assert dataset_name in ['mnist', 'fashion_mnist']

  print("Load training and eval data from tf.keras!!")
  # Load training and eval data from tf.keras
  if dataset_name == 'mnist':
    (train_images, train_labels), _ = \
      tf.keras.datasets.mnist.load_data()
  else:
    (train_images, train_labels), _ = \
      tf.keras.datasets.fashion_mnist.load_data()

  train_images_list = []
  train_labels_list = []
  test_images = []
  test_labels = []

  train_len = train_labels.shape[0]
  shuffled_index = list(range(train_len))
  random.seed(12345)
  random.shuffle(shuffled_index)
  train_images_list = [train_images[i] for i in shuffled_index]
  train_labels_list = [train_labels[i] for i in shuffled_index]
  train_images = np.array(train_images_list)
  train_labels = np.array(train_labels_list)
  print(f"train_labels : {train_labels}")

  test_len = train_labels.shape[0] // 1000
  train_images_list = []
  train_labels_list = []

  for idx, (image, label) in enumerate(zip(train_images, train_labels)):
    if test_len < idx:
      train_images_list.append(image)
      train_labels_list.append(label)
    else:
      test_images.append(image)
      test_labels.append(label)

  train_images = np.array(train_images_list)
  train_labels = np.array(train_labels_list)
  test_images = np.array(test_images)
  test_labels = np.array(test_labels)
  print(f"test_labels : {test_labels}")

  train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
  test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

  print("Compute anomaly scores!!")

  for idx, (test_image, test_label) in enumerate(zip(test_images, test_labels)):
    test_image = (test_image / 127.5) - 1
    test_image = np.expand_dims(test_image, axis=0)
    gen_output = generator(test_image, training=False)
    disc_real_output = discriminator([test_image, test_image], training=False)
    disc_generated_output = discriminator([test_image, gen_output], training=False)

    anomaly_score, con_loss, lat_loss, adv_loss = generator_loss(gen_output,
                   test_image,
                   disc_real_output,
                   disc_generated_output)
    generate_images("./generate_image/{}_idx_{}_anomaly_score_{}.jpg".format(test_label, idx, anomaly_score), generator, test_image, test_image)
