# Literature Review

## Papers

| S/N | Name | Category | Summary |
| :--- | :--- | :--- | :--- |
| 1 | Deep Residual Learning for Image Recognition | Architecture | Residual Networks make use of "shortcuts" connections which allow succeeding layers to build upon features learnt from previous layers. Shortcut connections also have the added benefit of allowing gradients to flow across different layers. |
| 2 | Deep Networks with Stochastic Depth | Architecture | Residual Networks with Stochastic Depth takes the idea of 'dropout' to residual blocks. Having the constant c in, x' = x + cf\(x\) being set to zero at random, with linearly-increasing probability as each deeper layers during training. The benefit is two-fold: Firstly, dropping out layers at time speed-up training of deep networks, and secondly, it allows layers that varying depth to communicate with each other.  |
| 3 | Densely Connected Convolutional Networks | Architecture | DenseNet identify that the trend of utilizing shortcut connections between early and later layers, which allows "better gradient and information flow". On top of that, previous studies hints of redunancy in resnet as dropping out random layers entirely will not affect performance. This paper takes the idea of shortcut connections to the extreme by proposing concentanation of all preceeding layers to the next layer. By doing so, every feature learnt can be directly used to build more advanced features. Hence, promoting feature reuse and eliminating redundancies. |
| 4 | Neural Ordinary Differential Equation | Architecture | Neural ODE shows that the equation for residual network, x' = x+f\(x\) is exactly euler's method x' = x + cf\(x\) when c = 1 for solving ODE. Since, Euler's method is a primitive ODE solver, the premise of this paper explored the use of modern ODE solver for training neural network, replacing resnet with an ODE. Hence, this paper opens up neural networks to centuries of ODE knowledge |
| 5 | An Overview of Multi-Task Learning in Deep Neural Networks | Architecture | Overview of multi-tasking in deep-learning. |
| 6 | Solving Multiclass Learning Problems via Error-Correcting Output Codes | ECOC | Breaking down multiclass classification problems into binary components. Neural networks is well-suited for binary tasks i.e. one-hot encoding. Following this paper from 1994, there are numerous ECOC encoding scheme that are extension of this idea. |
| 7 | Rich feature hierarchies for accurate object detection and semantic segmentation | Object Localization |  |
| 8 | Fast R-CNN | Object Localization |  |
| 9 | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Object Localization |  |
| 10 | Beyond one-hot encoding: Lower dimensional target embedding | ECOC | This paper propose an alternative to one-hot encoding, showing that we can lower the dimensionality from N to min of log\(N\) at no cost to accuracy. Experiments also shows using these alternative ECOC encoding speeds up convergence of training neural networks. |
| 11 | Self-Supervised GAN via Auxillary Rotation Loss | Self-Supervision | This paper claims that GANs is a non-stationary online environment which is prone to catastrophic forgetting, causing GANs to be cyclic and/or unstable during training. I think that the idea of self-supervision is interesting and is worthy of futher investigation |
| 12 | Progressive Growing of GANs for Improved Quality, Stability, and Variation |  | This paper propose an innovative way to train GANs i.e. by slowly increasing the image size with easing from 4x4, 8x8, until the network is able to produce full-resolution image. Paper also propose ways to "discourage unhealthy" competition between G & D by preventing G from reacting to escalations from G. The author argues that this way of training similar to multi-G / multi-D architecture. Methods to equalize learning rate was mentioned as large variance in it will cause LR to be both \(too high or too small\).  |
| 13 | Few-Shot Adversarial Learning of Realistic Neural Talking Head Model | Cool GAN |  |
| 14 | Noise2Noise: Learning Image Restoration without Clean Data |  Autoencoder | Amazing paper which applies the idea of zero-mean noise \(which image-stacking and long-exposure image relies on\) to train neural network. Training a denoiser using noisy images as both input & output, using a L2 loss will converge the network into its mean. Since, the mean is ZERO, the resultant output will be the denoised image |
| 15 | Deep Clustering for Unsupervised Learning of Visual Features | Autoencoder |  |
| 16 | Deep k-Means: Jointly clustering with k-Means and learning representations | Self-Supervision |  |
| 17 | Label-Removed Generative Adversarial Networks Incorporating with K-Means | Self-Supervision |  |
| 18 | Adversarial feature learning | Generative Adversarial Network |  |
| 19 | An Empirical Study of Generative Models with Encoders | Generative Adversarial Network |  |
| 20 | Progressive Growing of GANs for Improved Quality, Stability, and Variation | Generative Adversarial Network |  |
| 21 | Semantic Image Synthesis with Spatially-Adaptive Normalization | Generative Adversarial Network |  |
| 22 | Transferring GANs: generating images from limited data | Generative Adversarial Network |  |
| 23 | OCGAN: One-class Novelty Detection Using GANs with Constrained LatentRepresentations | Generative Adversarial Network |  |
| 24 | Clustergan: Latent space clustering in generative adversarial networks | Generative Adversarial Network |  |
| 25 | Balanced Self-Paced Learning for Generative Adversarial Clustering Network | Generative Adversarial Network |  |
| 26 | Conditional Adversarial Generative Flow for Controllable Image Synthesis | Generative Adversarial Network |  |
| 27 | Generative Dual Adversarial Network for Generalized Zero-shot Learning | Generative Adversarial Network |  |
| 28 | Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis | Generative Adversarial Network |  |
| 29 | Label-Noise Robust Generative Adversarial Networks | Generative Adversarial Network |  |
|  |  |  |  |

