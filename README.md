# PDRec
The source code is for the paper: [Plug-In Diffusion Model for Sequential Recommendation](https://arxiv.org/pdf/2401.02913.pdf) accepted in AAAI 2024 by Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin and Zhanhui Kang.

## Overview
This paper presents a novel Plug-In Diffusion Model for Recommendation (PDRec) framework, which employs the diffusion model as a flexible plugin to jointly take full advantage of the diffusion-generating user preferences on all items. Specifically, PDRec first infers the users' dynamic preferences on all items via a time-interval diffusion model and proposes a Historical Behavior Reweighting (HBR) mechanism to identify the high-quality behaviors and suppress noisy behaviors. In addition to the observed items, PDRec proposes a Diffusion-based Positive Augmentation (DPA) strategy to leverage the top-ranked unobserved items as the potential positive samples, bringing in informative and diverse soft signals to alleviate data sparsity. To alleviate the false negative sampling issue, PDRec employs Noise-free Negative Sampling (NNS) to select stable negative samples for ensuring effective model optimization.![_](./overall_structure.png)

## Dependencies
- Python 3.8.10
- PyTorch 1.12.0+cu102
- pytorch-lightning==1.6.5
- Torchvision==0.8.2
- Pandas==1.3.5
- Scipy==1.7.3

## Implementation of RealHNS
I am working on organizing the related datasets and codes, which will be released in 2-3 days.

## BibTeX
If you find this work useful for your research, please kindly cite PDRec by:
```
@inproceedings{PDRec,
  title={Plug-in Diffusion Model for Sequential Recommendation},
  author={Ma, Haokai and Xie, Ruobing and Meng, Lei and Chen, Xin and Zhang, Xu and Lin, Leyu and Kang, Zhanhui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Acknowledgement
The structure of this code is largely based on [DiffRec](https://github.com/YiyanXu/DiffRec) and [SASRec](https://github.com/pmixer/SASRec.pytorch). Thanks for these works.




