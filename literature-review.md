# Literature Review

## Papers

| Name | Category | Summary |
| :--- | :--- | :--- |
| Deep Residual Learning for Image Recognition | Architecture | Residual Networks make use of "shortcuts" connections which allow succeeding layers to build upon features learnt from previous layers. Shortcut connections also have the added benefit of allowing gradients to flow across different layers. |
| Deep Networks with Stochastic Depth | Architecture | Residual Networks with Stochastic Depth takes the idea of 'dropout' to residual blocks. Having the constant c in, x' = x + cf\(x\) being set to zero at random, with linearly-increasing probability as each deeper layers during training. The benefit is two-fold: Firstly, dropping out layers at time speed-up training of deep networks, and secondly, it allows layers that varying depth to communicate with each other.  |
| Densely Connected Convolutional Networks | Architecture | DenseNet identify that the trend of utilizing shortcut connections between early and later layers, which allows "better gradient and information flow". On top of that, previous studies hints of redunancy in resnet as dropping out random layers entirely will not affect performance. This paper takes the idea of shortcut connections to the extreme by proposing concentanation of all preceeding layers to the next layer. By doing so, every feature learnt can be directly used to build more advanced features. Hence, promoting feature reuse and eliminating the redundancies. |
| Neural Ordinary Differential Equation | Architecture |  |
| An Overview of Multi-Task Learning in Deep Neural Networks | Architecture |  |
| Solving Multiclass Learning Problems via Error-Correcting Output Codes | ECOC |  |
| Rich feature hierarchies for accurate object detection and semantic segmentation | Object Localization |  |
| Fast R-CNN | Object Localization |  |
| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Object Localization |  |
| Beyond one-hot encoding: Lower dimensional target embedding | ECOC |  |
| Self-Supervised GAN via Auxillary Rotation Loss | Self-Supervision |  |
| Few-Shot Adversarial Learning of Realistic Neural Talking Head Model | Cool Application |  |
| Noise2Noise: Learning Image Restoration without Clean Data |  Autoencoder |  |
| Deep Clustering for Unsupervised Learning of Visual Features | Autoencoder |  |
| Deep k-Means: Jointly clustering with k-Means and learning representations | Self-Supervision |  |
| Label-Removed Generative Adversarial Networks Incorporating with K-Means | Self-Supervision |  |
| Adversarial feature learning | Generative Adversarial Network |  |
| An Empirical Study of Generative Models with Encoders | Generative Adversarial Network |  |

