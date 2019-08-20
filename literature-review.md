# Literature Review

## Papers

### Domain Adaptation

<table>
  <thead>
    <tr>
      <th style="text-align:left">Name</th>
      <th style="text-align:left">Summary</th>
      <th style="text-align:left">File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left">Domain-Adversarial Training of Neural Networks</td>
      <td style="text-align:left">Interesting paper where a neural net is trained to ignore domain specific
        features, in other to apply knowledge learnt from a labelled domain to
        an unlabelled domain. I believe that no new knowledge is gained from the
        unlabeled source</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1505.07818.pdf">https://arxiv.org/pdf/1505.07818.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Analysis of Representations for Domain Adaptation</td>
      <td style="text-align:left">Theory and Math behind Domain Adaption from label source to unlabeled
        target domain.</td>
      <td style="text-align:left"><a href="https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation.pdf">https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">A theory of learning from different domains</td>
      <td style="text-align:left">
        <p>More theory that I don&apos;t understand.</p>
        <p></p>
        <p>Domain Adaption looks to be similar to transfer learning. However, the
          labeling are preserved. It seems to me that domain adaptation can enhance
          network performance on the testing set, if the images are available before
          hand.</p>
      </td>
      <td style="text-align:left"><a href="https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36364.pdf">https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36364.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
    </tr>
  </tbody>
</table>### Fourier Neural Network

| Name | Summary | File |
| :--- | :--- | :--- |
|  |  |  |

### Network Architecture

| Name | Summary | File |
| :--- | :--- | :--- |
| Deep Residual Learning for Image Recognition | Residual Networks make use of "shortcuts" connections which allow succeeding layers to build upon features learnt from previous layers. Shortcut connections also have the added benefit of allowing gradients to flow across different layers. | [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf) |
| Deep Networks with Stochastic Depth | Residual Networks with Stochastic Depth takes the idea of 'dropout' to residual blocks. Having the constant c in, x' = x + cf\(x\) being set to zero at random, with linearly-increasing probability as each deeper layers during training. The benefit is two-fold: Firstly, dropping out layers at time speed-up training of deep networks, and secondly, it allows layers that varying depth to communicate with each other.  | [https://arxiv.org/pdf/1603.09382.pdf](https://arxiv.org/pdf/1603.09382.pdf)\` |
| Densely Connected Convolutional Networks | DenseNet identify that the trend of utilizing shortcut connections between early and later layers, which allows "better gradient and information flow". On top of that, previous studies hints of redunancy in resnet as dropping out random layers entirely will not affect performance. This paper takes the idea of shortcut connections to the extreme by proposing concentanation of all preceeding layers to the next layer. By doing so, every feature learnt can be directly used to build more advanced features. Hence, promoting feature reuse and eliminating redundancies. | [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf) |
| Neural Ordinary Differential Equation | Neural ODE shows that the equation for residual network, x' = x+f\(x\) is exactly euler's method x' = x + cf\(x\) when c = 1 for solving ODE. Since, Euler's method is a primitive ODE solver, the premise of this paper explored the use of modern ODE solver for training neural network, replacing resnet with an ODE. Hence, this paper opens up neural networks to centuries of ODE knowledge | [https://arxiv.org/pdf/1806.07366.pdf](https://arxiv.org/pdf/1806.07366.pdf) |
| An Overview of Multi-Task Learning in Deep Neural Networks | Overview of multi-tasking in deep-learning, shares commonly used architecture soft/hard parameter sharing. Interesting survey. | [https://arxiv.org/pdf/1706.05098.pdf](https://arxiv.org/pdf/1706.05098.pdf) |

### Object Localization

| Name | Summary | File |
| :--- | :--- | :--- |
| Rich feature hierarchies for accurate object detection and semantic segmentation |  |  |
| Fast R-CNN |  |  |
| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks |  |  |

### Unsupervised Network

| Name | Summary | File |
| :--- | :--- | :--- |
| Deep Clustering for Unsupervised Learning of Visual Features | This paper propose an architecture that can be leverage on the unlabeled datasets to learn discriminant features. Instead, pseudo-labels from k-means is used to guide training of network. | [https://arxiv.org/pdf/1807.05520.pdf](https://arxiv.org/pdf/1807.05520.pdf) |
| Deep k-Means: Jointly clustering with k-Means and learning representations | Previous work alternates training of k-means and auto-encoder. Deep k-Means seek to remove the seam between these two algorithm, and join them to form a single framework. | [https://arxiv.org/pdf/1902.06938.pdf](https://arxiv.org/pdf/1902.06938.pdf) |
| Label-Removed Generative Adversarial Networks Incorporating with K-Means | This paper introduce a way to prevent model collapse by using K-Means generated labels to guide network training. Though labeled-data is used, this architecture is grouped under _unconditional GAN_, the labels are not being used to condition the output_._ | [https://arxiv.org/pdf/1902.06938.pdf](https://arxiv.org/pdf/1902.06938.pdf) |
| Unsupervised Visual Representation Learning by Context Prediction | This paper propose a interesting way for representation learning by training the network to predict the relative position of a source image to the target image. Experiment shows that once the authors removed so-called "trival" solutions that rely on low-level features, the network is able to start learning more advanced features. The author claims that theirs in the first example of unsupervised pre-training on a larger dataset could led to performance boost in a smaller dataset. | [https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf) |
| Unsupervised Representation Learning By Predicting Image Rotation | This paper propose a simple and, yet powerful approach that allows the learning of high-level semantic features, by predicting the rotational orientation. Compared to similar approaches, there is no need to additional pre-processing to avoid "trival" solutions | [https://arxiv.org/pdf/1803.07728.pdf](https://arxiv.org/pdf/1803.07728.pdf) |
| Colorful Image Colorization | The main focus of this paper is to use deep learning for multi-modal recolorisation of images. On the side, the author endeavor into a cross-channel encoder and using the features learnt for classification task. Thus fur, I observed that self-supervised techniques can be broadly grouped into three styles: 1\) hiding information away from learner,  2\) feature learning before & after data augmentation, and 3\) self-labelling  | [https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf) |

### Error-Correcting Output Codes

| Name | Summary | File |
| :--- | :--- | :--- |
| Beyond One-hot Encoding: lower dimensional target embedding | Our contribution is two fold: \(i\) We show that random projections of the label space are a valid tool to find such lower dimensional embeddings, boosting dramatically convergence rates at zero computational cost; and \(ii\) we propose a normalized eigenrepresentation of the class manifold that encodes the targets with minimal information loss, improving the accuracy of random projections encoding while enjoying the same convergence rates. | [https://arxiv.org/pdf/1806.10805.pdf](https://arxiv.org/pdf/1806.10805.pdf) |

### Auto-Encoder

| Name | Summary | File |
| :--- | :--- | :--- |
| Noise2Noise: Learning Image Restoration without Clean Data | Amazing paper which applies the idea of zero-mean noise \(which image-stacking and long-exposure image relies on\) to train neural network. Training a denoiser using noisy images as both input & output, using a L2 loss will converge the network into its mean. Since, the mean is ZERO, the resultant output will be the denoised image. | [https://arxiv.org/pdf/1803.04189.pdf](https://arxiv.org/pdf/1803.04189.pdf) |

### Generative Adversarial Network

| Name | Summary | File |
| :--- | :--- | :--- |
| Self-Supervised GANs via Auxiliary Rotation Loss | This paper claims that GANs is a non-stationary online environment which is prone to catastrophic forgetting, causing GANs to be cyclic and/or unstable during training. I think that the idea of self-supervision is interesting and is worthy of further investigation | [https://arxiv.org/pdf/1811.11212.pdf](https://arxiv.org/pdf/1811.11212.pdf) |
| Progressive Growing of GANs for Improved Quality, Stability, and Variation | This paper propose an innovative way to train GANs i.e. by slowly increasing the image size with easing from 4x4, 8x8, until the network is able to produce full-resolution image. Paper also propose ways to "discourage unhealthy" competition between G & D by preventing G from reacting to escalations from G. The author argues that this way of training similar to multi-G / multi-D architecture. Methods to equalize learning rate was mentioned as large variance in it will cause LR to be both \(too high or too small\).  | [https://arxiv.org/pdf/1710.10196.pdf](https://arxiv.org/pdf/1710.10196.pdf) |
| Few-Shot Adversarial Learning of Realistic Neural Talking Head Models |  This paper adopts the idea of meta-learning for synthesizing talking head of various actors, with few example images. The core idea is to conduct a lengthy training of the network on person-generic parameters, following by a few-shot transfer learning to person-specific parameters. The main limitation of this technique is the need for landmark adaption, as a different people have obviously different features. \(Don't understand technical details\) | [https://arxiv.org/pdf/1905.08233.pdf](https://arxiv.org/pdf/1905.08233.pdf) |
| Adversarial feature learning | Bi-directional GAN | [https://arxiv.org/pdf/1605.09782.pdf](https://arxiv.org/pdf/1605.09782.pdf) |
| An Empirical Study of Generative Models with Encoders |  |  |
| Semantic Image Synthesis with Spatially-Adaptive Normalization |  |  |
| Transferring GANs: generating images from limited data |  |  |
| OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations |  |  |
| Clustergan: Latent space clustering in generative adversarial networks |  |  |
| Balanced Self-Paced Learning for Generative Adversarial Clustering Network | Interesting architecture where a clusterer is introduced to GAN, whose role is the reverse of generator. G takes a random noise and generates an images, while C takes an image and maps it into a latent space. D would need to an additional vector i.e. fake mapping to differentiate real/fake image + correct/incorrect latent space mapping. This architecture is similar to bi-directional GAN. | [https://bit.ly/2KR5rP8](https://bit.ly/2KR5rP8) |
| Conditional Adversarial Generative Flow for Controllable Image Synthesis |  |  |
| Generative Dual Adversarial Network for Generalized Zero-shot Learning |  |  |
| Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis |  |  |
| Label-Noise Robust Generative Adversarial Networks |  |  |
| Adversarially Learned Inference |  |  |
| Large Scale Adversarial Representation Learning | BigBiGans | [https://arxiv.org/pdf/1907.02544.pdf](https://arxiv.org/pdf/1907.02544.pdf) |
| DS3L: Deep Self-Semi-Supervised Learning for Image Recognition |  | [https://arxiv.org/pdf/1905.13305.pdf](https://arxiv.org/pdf/1905.13305.pdf) |
| Collaborative Sampling in Generative Adversarial |  | [https://arxiv.org/pdf/1902.00813.pdf](https://arxiv.org/pdf/1902.00813.pdf) |
| Conditional image synthesis with auxiliary classifier GANs |  | [https://arxiv.org/pdf/1610.09585.pdf](https://arxiv.org/pdf/1610.09585.pdf) |
| A Survey of Unsupervised Deep Domain Adaptation |  | [https://arxiv.org/pdf/1812.02849.pdf](https://arxiv.org/pdf/1812.02849.pdf) |
| Generative Compression |  | [https://arxiv.org/abs/1703.01467](https://arxiv.org/abs/1703.01467) |
| Font Size: It Takes \(Only\) Two: Adversarial Generator-Encoder Networks |  | [https://arxiv.org/pdf/1704.02304.pdf](https://arxiv.org/pdf/1704.02304.pdf) |
| Inverting the Generator of a Generative Adversarial Network |  | [https://arxiv.org/pdf/1611.05644.pdf](https://arxiv.org/pdf/1611.05644.pdf) |
| Adversarial Feature Augmentation for Unsupervised Domain Adaptation |  | [https://bit.ly/31zYrx0](https://bit.ly/31zYrx0) |
| Learning hierarchical features from deep generative models |  | [https://arxiv.org/pdf/1702.08396.pdf](https://arxiv.org/pdf/1702.08396.pdf) |
| Semantic Image Synthesis via Adversarial Learning |  | [https://bit.ly/30iJtuQ](https://bit.ly/30iJtuQ) |
| Loss is its own Reward: Self-Supervision for Reinforcement Learning |  | [https://arxiv.org/pdf/1612.07307.pdf](https://arxiv.org/pdf/1612.07307.pdf) |
| Structured Generative Adversarial Networks |  | [https://arxiv.org/pdf/1711.00889.pdf](https://arxiv.org/pdf/1711.00889.pdf) |
| Conditional Image-to-Image Translation |  | [https://arxiv.org/pdf/1805.00251.pdf](https://arxiv.org/pdf/1805.00251.pdf) |
| Ensembles of Generative Adversarial Networks |  | [https://arxiv.org/pdf/1612.00991.pdf](https://arxiv.org/pdf/1612.00991.pdf) |
| Denoising Adversarial Autoencoders |  | [https://arxiv.org/pdf/1703.01220.pdf](https://arxiv.org/pdf/1703.01220.pdf) |
| How Generative Adversarial Networks and Their Variants Work: An Overview |  | [https://arxiv.org/pdf/1711.05914.pdf](https://arxiv.org/pdf/1711.05914.pdf) |
| Self-Supervised Feature Learning by Learning to Spot Artifacts |  | [https://arxiv.org/pdf/1806.05024.pdf](https://arxiv.org/pdf/1806.05024.pdf) |
| Lifelong Generative Modeling |  | [https://arxiv.org/pdf/1705.09847.pdf](https://arxiv.org/pdf/1705.09847.pdf) |
| Generative Adversarial Image Synthesis With Decision Tree Latent Controller |  | [https://arxiv.org/pdf/1805.10603.pdf](https://arxiv.org/pdf/1805.10603.pdf) |
| Bidirectional Conditional Generative Adversarial Networks |  | [https://arxiv.org/pdf/1711.07461.pdf](https://arxiv.org/pdf/1711.07461.pdf) |
| Stacked Generative Adversarial Networks |  | [https://arxiv.org/pdf/1612.04357.pdf](https://arxiv.org/pdf/1612.04357.pdf) |
| Neural Photo Editing with Introspective Adversarial Networks |  | [https://arxiv.org/pdf/1609.07093.pdf](https://arxiv.org/pdf/1609.07093.pdf) |
| Adversarial Discriminative Domain Adaptation |  | [https://arxiv.org/pdf/1702.05464.pdf](https://arxiv.org/pdf/1702.05464.pdf) |
| Controllable Generative Adversarial Network |  | [https://arxiv.org/pdf/1708.00598.pdf](https://arxiv.org/pdf/1708.00598.pdf) |
|  |  |  |

