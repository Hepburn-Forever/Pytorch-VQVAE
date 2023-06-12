VQ-VAE是一种图像压缩和重构模型，它可以通过学习一个可以组合不同子部分的嵌入空间来实现图像的高效压缩，同时保留图像的细节。

VQ-VAE中的VQ代表向量量化，VAE代表变分自编码器。它由两个子模型组成：

1.编码器：它将输入图像作为输入，并生成一个由向量组成的离散潜在变量编码。

2.解码器：它将离散潜在变量编码作为输入，并生成重构图像。

在训练阶段，VQ-VAE通过最小化重构误差来优化其参数。在生成阶段，它可以通过指定一个离散潜在变量编码来生成一个图像，并且可以通过将不同的编码组合在一起来生成不同的图像。

总的来说，VQ-VAE是一种用于压缩和重构图像的强大工具，可以通过学习一个可以组合不同子部分的嵌入空间来实现高效的压缩，并且可以通过离散编码来实现更好的泛化能力。

## Reproducing Neural Discrete Representation Learning
### Course Project for [IFT 6135 - Representation Learning](https://ift6135h18.wordpress.com/)

Project Report link: [final_project.pdf](final_project.pdf)

### Instructions
1. To train the VQVAE with default arguments as discussed in the report, execute:
```
python vqvae.py --data-folder /tmp/miniimagenet --output-folder models/vqvae
```
2. To train the PixelCNN prior on the latents, execute:
```
python pixelcnn_prior.py --data-folder /tmp/miniimagenet --model models/vqvae --output-folder models/pixelcnn_prior
```
### Datasets Tested
#### Image
1. MNIST
2. FashionMNIST
3. CIFAR10
4. Mini-ImageNet

#### Video
1. Atari 2600 - Boxing (OpenAI Gym) [code](https://github.com/ritheshkumar95/pytorch-vqvae/tree/evan/video)

### Reconstructions from VQ-VAE
Top 4 rows are Original Images. Bottom 4 rows are Reconstructions.
#### MNIST
![png](samples/vqvae_reconstructions_MNIST.png)
#### Fashion MNIST
![png](samples/vqvae_reconstructions_FashionMNIST.png)

### Class-conditional samples from VQVAE with PixelCNN prior on the latents
#### MNIST
![png](samples/samples_MNIST.png)
#### Fashion MNIST
![png](samples/samples_FashionMNIST.png)

### Comments
1. We noticed that implementing our own VectorQuantization PyTorch function speeded-up training of VQ-VAE by nearly 3x. The slower, but simpler code is in this [commit](https://github.com/ritheshkumar95/pytorch-vqvae/tree/cde142670f701e783f29e9c815f390fc502532e8).
2. We added some basic tests for the vector quantization functions (based on `pytest`). To run these tests
```
py.test . -vv
```

### Authors
1. Rithesh Kumar
2. Tristan Deleu
3. Evan Racah
