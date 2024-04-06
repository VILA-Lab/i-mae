# A PyTorch Implementation of i-MAE: Linearly Separable Representation in MAE

[Kevin Zhang*](https://kzyz.netlify.com/), [Zhiqiang Shen*](http://zhiqiangshen.com/)

[`Project Page`](https://zhiqiangshen.com/projects/i-mae/)
| [`Paper`](https://arxiv.org/abs/2210.11470)
| [`BibTeX`](#citation)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vision-learning-acceleration-lab/i-mae/blob/main/i-MAE_demo.ipynb)

![i-MAE2](https://user-images.githubusercontent.com/52997677/196735725-496592d3-5883-4db4-ba34-04d0a8dab535.svg)

We provide a PyTorch/GPU based implementation of our technical report [i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable?
](https://arxiv.org/abs/2210.11470)

### Catalog

- [X] Pretrain demo with Colab
- [X] Pre-training and Fine-tuning code
- [ ] Weights Upload

### Pre-training

The pre-training instruction is in [PRETRAIN.md](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md).

### Fine-tuning

The fine-tuning instruction is in [FINETUNE.md](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md).

### Visualization demo

Please visit our interactive demo on our [website](https://zhiqiangshen.com/projects/i-mae/), or run our visualization demo with a [Colab notebook](https://colab.research.google.com/github/vision-learning-acceleration-lab/i-mae/blob/main/i-MAE_demo.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vision-learning-acceleration-lab/i-mae/blob/main/i-MAE_demo.ipynb)

### Acknowledgement

This repository is based on [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm) and [MAE](https://github.com/facebookresearch/mae) repositories.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation

If you find this repository helpful, please consider citing our work:

```
@article{zhang2022i-mae,
  title={i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable?},
  author = {Zhang, Kevin and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2210.11470},
  year={2022}
}
```
