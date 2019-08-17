# Literature Review

## Papers

| Name | Category | Summary | File |
| :--- | :--- | :--- | :--- |
| Deep Residual Learning for Image Recognition | Architecture | Residual Networks make use of "shortcuts" connections which allow succeeding layers to build upon features learnt from previous layers. Shortcut connections also have the added benefit of allowing gradients to flow across different layers. | [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf) |
| Deep Networks with Stochastic Depth | Architecture | Residual Networks with Stochastic Depth takes the idea of 'dropout' to residual blocks. Having the constant c in, x' = x + cf\(x\) being set to zero at random, with linearly-increasing probability as each deeper layers during training. The benefit is two-fold: Firstly, dropping out layers at time speed-up training of deep networks, and secondly, it allows layers that varying depth to communicate with each other.  | [https://arxiv.org/pdf/1603.09382.pdf](https://arxiv.org/pdf/1603.09382.pdf) |
| Densely Connected Convolutional Networks | Architecture | DenseNet identify that the trend of utilizing shortcut connections between early and later layers, which allows "better gradient and information flow". On top of that, previous studies hints of redunancy in resnet as dropping out random layers entirely will not affect performance. This paper takes the idea of shortcut connections to the extreme by proposing concentanation of all preceeding layers to the next layer. By doing so, every feature learnt can be directly used to build more advanced features. Hence, promoting feature reuse and eliminating redundancies. | [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf) |
| Neural Ordinary Differential Equation | Architecture | Neural ODE shows that the equation for residual network, x' = x+f\(x\) is exactly euler's method x' = x + cf\(x\) when c = 1 for solving ODE. Since, Euler's method is a primitive ODE solver, the premise of this paper explored the use of modern ODE solver for training neural network, replacing resnet with an ODE. Hence, this paper opens up neural networks to centuries of ODE knowledge | [https://arxiv.org/pdf/1806.07366.pdf](https://arxiv.org/pdf/1806.07366.pdf) |
| An Overview of Multi-Task Learning in Deep Neural Networks | Architecture | Overview of multi-tasking in deep-learning, shares commonly used architecture soft/hard parameter sharing. Interesting survey. | [https://arxiv.org/pdf/1706.05098.pdf](https://arxiv.org/pdf/1706.05098.pdf) |
| Solving Multiclass Learning Problems via Error-Correcting Output Codes | ECOC | Breaking down multiclass classification problems into binary components. Neural networks is well-suited for binary tasks i.e. one-hot encoding. Following this paper from 1994, there are numerous ECOC encoding scheme that are extension of this idea. | [https://arxiv.org/pdf/cs/9501101.pdf](https://arxiv.org/pdf/cs/9501101.pdf) |
| Rich feature hierarchies for accurate object detection and semantic segmentation | Object Localization |  |  |
| Fast R-CNN | Object Localization |  |  |
| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Object Localization |  |  |
| Beyond one-hot encoding: Lower dimensional target embedding | ECOC | This paper propose an alternative to one-hot encoding, showing that we can lower the dimensionality from N to min of log\(N\) at no cost to accuracy. Experiments also shows using these alternative ECOC encoding speeds up convergence of training neural networks. | [https://arxiv.org/pdf/1806.10805.pdf](https://arxiv.org/pdf/1806.10805.pdf) |
| Self-Supervised GANs via Auxiliary Rotation Loss | Self-Supervision | This paper claims that GANs is a non-stationary online environment which is prone to catastrophic forgetting, causing GANs to be cyclic and/or unstable during training. I think that the idea of self-supervision is interesting and is worthy of futher investigation | [https://arxiv.org/pdf/1811.11212.pdf](https://arxiv.org/pdf/1811.11212.pdf) |
| Progressive Growing of GANs for Improved Quality, Stability, and Variation | Generative Adversarial Network | This paper propose an innovative way to train GANs i.e. by slowly increasing the image size with easing from 4x4, 8x8, until the network is able to produce full-resolution image. Paper also propose ways to "discourage unhealthy" competition between G & D by preventing G from reacting to escalations from G. The author argues that this way of training similar to multi-G / multi-D architecture. Methods to equalize learning rate was mentioned as large variance in it will cause LR to be both \(too high or too small\).  | [https://arxiv.org/pdf/1710.10196.pdf](https://arxiv.org/pdf/1710.10196.pdf) |
| Few-Shot Adversarial Learning of Realistic Neural Talking Head Model | Cool GAN |  | [https://arxiv.org/pdf/1905.08233.pdf](https://arxiv.org/pdf/1905.08233.pdf) |
| Noise2Noise: Learning Image Restoration without Clean Data |  Autoencoder | Amazing paper which applies the idea of zero-mean noise \(which image-stacking and long-exposure image relies on\) to train neural network. Training a denoiser using noisy images as both input & output, using a L2 loss will converge the network into its mean. Since, the mean is ZERO, the resultant output will be the denoised image. | [https://arxiv.org/pdf/1803.04189.pdf](https://arxiv.org/pdf/1803.04189.pdf) |
| Deep Clustering for Unsupervised Learning of Visual Features | Autoencoder |  |  |
| Deep k-Means: Jointly clustering with k-Means and learning representations | Self-Supervision |  | [https://arxiv.org/pdf/1902.06938.pdf](https://arxiv.org/pdf/1902.06938.pdf) |
| Label-Removed Generative Adversarial Networks Incorporating with K-Means | Self-Supervision | This paper introduce a way to prevent model collapse by using K-Means generated labels to guide network training. Though labeled-data is used, this architecture is grouped under _unconditional GAN_, the labels are not being used to condition the output_._ | [https://arxiv.org/pdf/1902.06938.pdf](https://arxiv.org/pdf/1902.06938.pdf) |
| Adversarial feature learning | Generative Adversarial Network | Bi-directional GAN | [https://arxiv.org/pdf/1605.09782.pdf](https://arxiv.org/pdf/1605.09782.pdf) |
| An Empirical Study of Generative Models with Encoders | Generative Adversarial Network |  |  |
| Semantic Image Synthesis with Spatially-Adaptive Normalization | Generative Adversarial Network |  |  |
| Transferring GANs: generating images from limited data | Generative Adversarial Network |  |  |
| OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations | Generative Adversarial Network |  |  |
| Clustergan: Latent space clustering in generative adversarial networks | Generative Adversarial Network |  |  |
| Balanced Self-Paced Learning for Generative Adversarial Clustering Network | Generative Adversarial Network | Interesting architecture where a clusterer is introduced to GAN, whose role is the reverse of generator. G takes a random noise and generates an images, while C takes an image and maps it into a latent space. D would need to an additional vector i.e. fake mapping to differentiate real/fake image + correct/incorrect latent space mapping. This architecture is similar to bi-directional GAN. | [https://bit.ly/2KR5rP8](https://bit.ly/2KR5rP8) |
| Conditional Adversarial Generative Flow for Controllable Image Synthesis | Generative Adversarial Network |  |  |
| Generative Dual Adversarial Network for Generalized Zero-shot Learning | Generative Adversarial Network |  |  |
| Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis | Generative Adversarial Network |  |  |
| Label-Noise Robust Generative Adversarial Networks | Generative Adversarial Network |  |  |
| Adversarially Learned Inference | Generative Adversarial Network |  |  |
| Large Scale Adversarial Representation Learning | Generative Adversarial Network | BigBiGans | [https://arxiv.org/pdf/1907.02544.pdf](https://arxiv.org/pdf/1907.02544.pdf) |
| DS3L: Deep Self-Semi-Supervised Learning for Image Recognition |  |  | [https://arxiv.org/pdf/1905.13305.pdf](https://arxiv.org/pdf/1905.13305.pdf) |
| Collaborative Sampling in Generative Adversaria |  |  | [https://arxiv.org/pdf/1902.00813.pdf](https://arxiv.org/pdf/1902.00813.pdf) |
| Conditional image synthesis with auxiliary classifier GANs |  |  | [https://arxiv.org/pdf/1610.09585.pdf](https://arxiv.org/pdf/1610.09585.pdf) |
| A Survey of Unsupervised Deep Domain Adaptation |  |  | [https://arxiv.org/pdf/1812.02849.pdf](https://arxiv.org/pdf/1812.02849.pdf) |
| Generative Compression |  |  | [https://arxiv.org/abs/1703.01467](https://arxiv.org/abs/1703.01467) |
| Font Size: It Takes \(Only\) Two: Adversarial Generator-Encoder Networks |  |  | [https://arxiv.org/pdf/1704.02304.pdf](https://arxiv.org/pdf/1704.02304.pdf) |
| Inverting the Generator of a Generative Adversarial Network |  |  | [https://arxiv.org/pdf/1611.05644.pdf](https://arxiv.org/pdf/1611.05644.pdf) |
| Adversarial Feature Augmentation for Unsupervised Domain Adaptation |  |  | [https://bit.ly/31zYrx0](https://bit.ly/31zYrx0) |
| Learning hierarchical features from deep generative models |  |  | [https://arxiv.org/pdf/1702.08396.pdf](https://arxiv.org/pdf/1702.08396.pdf) |
| Semantic Image Synthesis via Adversarial Learning |  |  | [https://bit.ly/30iJtuQ](https://bit.ly/30iJtuQ) |
| Loss is its own Reward: Self-Supervision for Reinforcement Learning |  |  | [https://arxiv.org/pdf/1612.07307.pdf](https://arxiv.org/pdf/1612.07307.pdf) |
| Structured Generative Adversarial Networks |  |  | [https://arxiv.org/pdf/1711.00889.pdf](https://arxiv.org/pdf/1711.00889.pdf) |
| Conditional Image-to-Image Translation |  |  | [https://arxiv.org/pdf/1805.00251.pdf](https://arxiv.org/pdf/1805.00251.pdf) |
| Ensembles of Generative Adversarial Networks |  |  | [https://arxiv.org/pdf/1612.00991.pdf](https://arxiv.org/pdf/1612.00991.pdf) |
| Denoising Adversarial Autoencoders |  |  | [https://arxiv.org/pdf/1703.01220.pdf](https://arxiv.org/pdf/1703.01220.pdf) |
| How Generative Adversarial Networks and Their Variants Work: An Overview |  |  | [https://arxiv.org/pdf/1711.05914.pdf](https://arxiv.org/pdf/1711.05914.pdf) |
| Self-Supervised Feature Learning by Learning to Spot Artifacts |  |  | [https://arxiv.org/pdf/1806.05024.pdf](https://arxiv.org/pdf/1806.05024.pdf) |
| Lifelong Generative Modeling |  |  | [https://arxiv.org/pdf/1705.09847.pdf](https://arxiv.org/pdf/1705.09847.pdf) |
| Generative Adversarial Image Synthesis With Decision Tree Latent Controller |  |  | [https://arxiv.org/pdf/1805.10603.pdf](https://arxiv.org/pdf/1805.10603.pdf) |
| Bidirectional Conditional Generative Adversarial Networks |  |  | [https://arxiv.org/pdf/1711.07461.pdf](https://arxiv.org/pdf/1711.07461.pdf) |
| Stacked Generative Adversarial Networks |  |  | [https://arxiv.org/pdf/1612.04357.pdf](https://arxiv.org/pdf/1612.04357.pdf) |
| Neural Photo Editing with Introspective Adversarial Networks |  |  | [https://arxiv.org/pdf/1609.07093.pdf](https://arxiv.org/pdf/1609.07093.pdf) |
| Adversarial Discriminative Domain Adaptation |  |  | [https://arxiv.org/pdf/1702.05464.pdf](https://arxiv.org/pdf/1702.05464.pdf) |
| Controllable Generative Adversarial Network |  |  | [https://arxiv.org/pdf/1708.00598.pdf](https://arxiv.org/pdf/1708.00598.pdf) |

