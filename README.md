# FFA-Net-Keras

This is the code repository for my best attempt in coding the FFA-Net dehazing architecture (Qin et al.: "FFA-Net: Feature Fusion Attention Network for Single Image Dehazing") in Keras, as so far there are no code repository dedicated to this, and the original work utilized PyTorch. The architecture of the FFA-Net is illustrated below and is straight from the paper.

Hopefully this repository would be useful for anyone who is researching/learning image dehazing but prefer working in the tensorflow/keras language. 

![FFA](https://github.com/user-attachments/assets/abb8c241-eca5-4939-9e8c-6630672d1862)


Link to the paper: https://arxiv.org/pdf/1911.07559 (ArXiV) 

Link to the paper: https://cdn.aaai.org/ojs/6865/6865-13-10094-1-10-20200525.pdf (AAAI Format)

Original github link for the work (in PyTorch): https://github.com/zhilin007/FFA-Net

Please let me know if you have any questions. All the best in your research and/or learning.

# Dependencies

-python3

-keras version 3.6.0

-Any GPUs or TPUs

-numpy

-matplotlib

# Brief Instructions 

- Load the hazy and clear image datasets of your choice.
- Run FFA-Net.py (containing the main architectures of the FFA-Net).
- Run Learning_Config.py (containing the cosine learning schedule and the Adam SGD, with the parameters adjusted to stick to the paper's config as close as possible).
- Run training_and_prediction.py for the training and testing on your test hazy images.
- Finally, run PSNR_and_SSIM.py to evaluate the PSNR and SSIM values of your dehazed images relative to the ground-truth images.


# Original Paper References
[1] X. Qin, Z. Wang, Y. Bai, X. Xie, and H. Jia, “Ffa-net: Feature fusion attention network for single image dehazing,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 11 908–11 915. 48, 99, 107.
