# skip ganormaly

## Network
![skip-ganormaly network](./images/skipgan2.png)

## Train
```bash
$ bash skip-ganormaly.sh
```

## Tensorboard
```bash
$ tensorboard --logdir ./log
```

## Result
```bash
$ ls ./output
epoch_0_0.jpg
....
```

## Result Image
![fashin mnist](./images/0_idx_280_anomaly_score_2.535538911819458_0.jpg)

## Inference
- label number : 0
- idx : 16 
- anomaly_score : 1.7745494842529297
```bash
$ ls ./generate_image
0_idx_16_anomaly_score_1.7745494842529297_0.jpg
....
```

### Paper
- [Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection](https://arxiv.org/abs/1901.08954)

### Reference
 - https://nbviewer.jupyter.org/github/ilguyi/generative.models.tensorflow.v2/blob/master/gans/dcgan.ipynb
 - https://github.com/samet-akcay/skip-ganomaly
 - https://www.tensorflow.org/tutorials/generative/pix2pix
