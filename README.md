# Rethinking Identity Mapping in Self-Supervised Hyperspectral Anomaly Detection: A Unified Perspective on Network Optimization


This is the official pytorch implementation of Rethinking Identity Mapping in Self-Supervised Hyperspectral Anomaly Detection: A Unified Perspective on Network Optimization (Undergoing Review).

Code will be released after peer review.

<hr />

> **Abstract:** *The surge of deep learning has catalyzed considerable progress in self-supervised Hyperspectral Anomaly Detection (HAD). The core premise for self-supervised HAD is that anomalous pixels are inherently more challenging to reconstruct, resulting in larger errors compared to the background. However, owing to the powerful nonlinear fitting capabilities of neural networks, self-supervised models often suffer from the Identity Mapping Problem (IMP). The IMP manifests as a tendency for the model to overfit to the entire image, particularly with increasing network complexity or prolonged training iterations. Consequently, the whole image can be precisely reconstructed, and even the anomalous pixels exhibit imperceptible errors, making them difficult to detect. Despite the proposal of several models aimed at addressing the IMP-related issues, a unified descriptive framework and validation of solutions for IMP remain lacking. In this paper, we conduct an in-depth exploration to IMP, and summarize a unified framework that describes IMP from the perspective of network optimization, which encompasses three aspects: perturbation, reconstruction, and regularization. Correspondingly, we introduce three solutions: superpixel pooling and upooling for perturbation, error-adaptive convolution for reconstruction, and online background pixel mining for regularization. With extensive experiments being conducted to validate the effectiveness, it is hoped that our work will provide valuable insights and inspire further research for self-supervised HAD. Code: https://github.com/yc-cui/Super-AD.*
<hr />


## Network Architecture

![](assets/overview.png)
