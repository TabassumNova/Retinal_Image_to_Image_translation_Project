# Retinal_Image_translation_project
Digital Fundus Image (DFI) and Fluorescein Angiography (FA) are two popular modalities of choice for retinal diagnostic. In this project, I used CycleGAN with Shape-consistency loss proposed by Zhang et al.[[1](https://arxiv.org/abs/1802.09655)] to
perform image-to-image translation between these two modalities.

<img src="/images/model.png"  width=70% height=70%>

# Requirements
- torch
- numpy
- opencv-python
- imageio
- albumentations
- gdown
- glob
- tqdm
- torchvision
- dominate
- visdom

# Result after Segmentation
For segmentator model, I used the implementation of [[3](https://github.com/srijarkoroy/segRetino)]

<img src="/images/seg.png"  width=50% height=50%>

# Result of Image-to-Image translation
<img src="/images/result1.png"  width=50% height=50%>
<img src="/images/result2.png"  width=50% height=50%>

# References
1. https://arxiv.org/abs/1802.09655
2. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
3. https://github.com/srijarkoroy/segRetino
