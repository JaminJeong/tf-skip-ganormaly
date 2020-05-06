# gray-to-color

## pix2pix

## Data Augmentation
```bash
$ bash data_augmentation.sh
```
## Train
```bash
$ bash pix2pix-gray2color.sh
```

## Tensorboard
```bash
$ tensorboard --logdir ./log
```

## Result
```bash
$ ls output
epoch_0.jpg
....
epoch_100.jpg

```
![flower_gray_origin_generation](./images/epoch_148.jpg)


## test
```bash
$ bash inference.sh
$ ls generate_image

```

![human_gray_origin_generation](./images/result.jpg)


### Reference
 - https://www.tensorflow.org/tutorials/generative/pix2pix
 - https://phillipi.github.io/pix2pix/
 - https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/dcgan.ipynb
