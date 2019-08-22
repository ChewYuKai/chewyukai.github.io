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
| Unsupervised Representation Learning By Predicting Image Rotation | This paper propose a simple and, yet powerful approach that allows the learning of high-level semantic features, by predicting the rotational orientation. Compared to similar approaches, there is no need to additional pre-processing to avoid "trivial" solutions | [https://arxiv.org/pdf/1803.07728.pdf](https://arxiv.org/pdf/1803.07728.pdf) |
| Colorful Image Colorization | The main focus of this paper is to use deep learning for multi-modal re-colorisation of images. On the side, the author endeavor into a cross-channel encoder and using the features learnt for classification task. Thus fur, I observed that self-supervised techniques can be broadly grouped into three styles: 1\) hiding information away from learner,  2\) feature learning before & after data augmentation, and 3\) self-labelling  | [https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf) |

### Error-Correcting Output Codes

| Name | Summary | File |
| :--- | :--- | :--- |
| Beyond One-hot Encoding: lower dimensional target embedding | Our contribution is two fold: \(i\) We show that random projections of the label space are a valid tool to find such lower dimensional embeddings, boosting dramatically convergence rates at zero computational cost; and \(ii\) we propose a normalized eigenrepresentation of the class manifold that encodes the targets with minimal information loss, improving the accuracy of random projections encoding while enjoying the same convergence rates. | [https://arxiv.org/pdf/1806.10805.pdf](https://arxiv.org/pdf/1806.10805.pdf) |

### Auto-Encoder

| Name | Summary | File |
| :--- | :--- | :--- |
| Noise2Noise: Learning Image Restoration without Clean Data | Amazing paper which applies the idea of zero-mean noise \(which image-stacking and long-exposure image relies on\) to train neural network. Training a de-noiser using noisy images as both input & output, using a L2 loss will converge the network into its mean. Since, the mean is ZERO, the resultant output will be the denoised image. | [https://arxiv.org/pdf/1803.04189.pdf](https://arxiv.org/pdf/1803.04189.pdf) |

### Generative Adversarial Network

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
      <td style="text-align:left">Self-Supervised GANs via Auxiliary Rotation Loss</td>
      <td style="text-align:left">This paper claims that GANs is a non-stationary online environment which
        is prone to catastrophic forgetting, causing GANs to be cyclic and/or unstable
        during training. I think that the idea of self-supervision is interesting
        and is worthy of further investigation</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1811.11212.pdf">https://arxiv.org/pdf/1811.11212.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Progressive Growing of GANs for Improved Quality, Stability, and Variation</td>
      <td
      style="text-align:left">This paper propose an innovative way to train GANs i.e. by slowly increasing
        the image size with easing from 4x4, 8x8, until the network is able to
        produce full-resolution image. Paper also propose ways to &quot;discourage
        unhealthy&quot; competition between G &amp; D by preventing G from reacting
        to escalations from G. The author argues that this way of training similar
        to multi-G / multi-D architecture. Methods to equalize learning rate was
        mentioned as large variance in it will cause LR to be both (too high or
        too small).</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1710.10196.pdf">https://arxiv.org/pdf/1710.10196.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Few-Shot Adversarial Learning of Realistic Neural Talking Head Models</td>
      <td
      style="text-align:left">This paper adopts the idea of meta-learning for synthesizing talking head
        of various actors, with few example images. The core idea is to conduct
        a lengthy training of the network on person-generic parameters, following
        by a few-shot transfer learning to person-specific parameters. The main
        limitation of this technique is the need for landmark adaption, as a different
        people have obviously different features. I believe that the few-shot capability
        of this network, is constrained to only talking head models due to the
        fundamental limitation of the applied meta-learning.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.08233.pdf">https://arxiv.org/pdf/1905.08233.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial feature learning</td>
      <td style="text-align:left">Bi-directional GAN</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1605.09782.pdf">https://arxiv.org/pdf/1605.09782.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Semantic Image Synthesis with Spatially-Adaptive Normalization</td>
      <td
      style="text-align:left">The innovation of this paper is the introduction of SPADE, which produce
        better quality images from semantic representation. The innovation seems
        to be problem specific.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.07291.pdf">https://arxiv.org/pdf/1903.07291.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Transferring GANs: generating images from limited data</td>
      <td style="text-align:left">This paper is an empirical study of transfer-learning/domain-adaptation
        of GANS. Comparison have been made between pre-trained vs randomly-initilized
        network.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.01677.pdf">https://arxiv.org/pdf/1805.01677.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">OCGAN: One-class Novelty Detection Using GANs with Constrained Latent
        Representations</td>
      <td style="text-align:left">In representation learning, the focus is to preserve the details, and
        assuming out-of-class objects follows the same representation logic. The
        main focus on this paper is not only ensuring that in-class objects are
        well-represented, but also that out-of-class objects are poorly represented.
        The author proposed negative-information mining as a means to detect out-of-class
        cases without actual negative examples.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.08550.pdf">https://arxiv.org/pdf/1903.08550.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Clustergan: Latent space clustering in generative adversarial networks</td>
      <td
      style="text-align:left">This paper is the first that I have seen that suggest that cluster structure
        is not preserved in GANs. The authors explored different ways to present
        the priors such that, clustering structure is preserved. Not very interesting,
        seems like conditional GAN.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1809.03627.pdf">https://arxiv.org/pdf/1809.03627.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Balanced Self-Paced Learning for Generative Adversarial Clustering Network</td>
      <td
      style="text-align:left">Interesting architecture where a clusterer is introduced to GAN, whose
        role is the reverse of generator. G takes a random noise and generates
        an images, while C takes an image and maps it into a latent space. D would
        need to an additional vector i.e. fake mapping to differentiate real/fake
        image + correct/incorrect latent space mapping. This architecture is similar
        to bi-directional GAN.</td>
        <td style="text-align:left"><a href="https://bit.ly/2KR5rP8">https://bit.ly/2KR5rP8</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional Adversarial Generative Flow for Controllable Image Synthesis</td>
      <td
      style="text-align:left">This paper explores the conditioning for flow-based adversarial generator.
        This may be useful of flow-based generator becomes a research focus.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1904.01782.pdf">https://arxiv.org/pdf/1904.01782.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Dual Adversarial Network for Generalized Zero-shot Learning</td>
      <td
      style="text-align:left">This paper propose to use cycle-consistency from image features-to-class
        labels, for the purpose of zero shot learning. Interesting application
        of cycleGANs concept beyond image-to-image translation. Similar examples
        cited in this paper that involves dual-learning like language translation,
        English -&gt; French &amp; French -&gt; English.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1811.04857.pdf">https://arxiv.org/pdf/1811.04857.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis</td>
      <td
      style="text-align:left">
        <p>This paper propose a way to prevent modal collapse in cGANs with a regularization
          term, dependent on the distance between the latent codes. Distant codes
          should produce very different image compared to close-by codes. The challenge
          of this approach is to preserve image realism and that the generator continues
          to present the real data distribution, as we seek to maximize diversity.
          The work experiment on three types of conditioning namely</p>
        <p>categorical, image-to-image, and text-to-image.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.05628.pdf">https://arxiv.org/pdf/1903.05628.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Label-Noise Robust Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarially Learned Inference</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
    </tr>
    <tr>
      <td style="text-align:left">Large Scale Adversarial Representation Learning</td>
      <td style="text-align:left">BigBiGans</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.02544.pdf">https://arxiv.org/pdf/1907.02544.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">DS3L: Deep Self-Semi-Supervised Learning for Image Recognition</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.13305.pdf">https://arxiv.org/pdf/1905.13305.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Collaborative Sampling in Generative Adversarial</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1902.00813.pdf">https://arxiv.org/pdf/1902.00813.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional image synthesis with auxiliary classifier GANs</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1610.09585.pdf">https://arxiv.org/pdf/1610.09585.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">A Survey of Unsupervised Deep Domain Adaptation</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1812.02849.pdf">https://arxiv.org/pdf/1812.02849.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Compression</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/abs/1703.01467">https://arxiv.org/abs/1703.01467</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Font Size: It Takes (Only) Two: Adversarial Generator-Encoder Networks</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1704.02304.pdf">https://arxiv.org/pdf/1704.02304.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Inverting the Generator of a Generative Adversarial Network</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1611.05644.pdf">https://arxiv.org/pdf/1611.05644.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Feature Augmentation for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://bit.ly/31zYrx0">https://bit.ly/31zYrx0</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Learning hierarchical features from deep generative models</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1702.08396.pdf">https://arxiv.org/pdf/1702.08396.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Semantic Image Synthesis via Adversarial Learning</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://bit.ly/30iJtuQ">https://bit.ly/30iJtuQ</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Loss is its own Reward: Self-Supervision for Reinforcement Learning</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1612.07307.pdf">https://arxiv.org/pdf/1612.07307.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Structured Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.00889.pdf">https://arxiv.org/pdf/1711.00889.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional Image-to-Image Translation</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.00251.pdf">https://arxiv.org/pdf/1805.00251.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Ensembles of Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1612.00991.pdf">https://arxiv.org/pdf/1612.00991.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Denoising Adversarial Autoencoders</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1703.01220.pdf">https://arxiv.org/pdf/1703.01220.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">How Generative Adversarial Networks and Their Variants Work: An Overview</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.05914.pdf">https://arxiv.org/pdf/1711.05914.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Self-Supervised Feature Learning by Learning to Spot Artifacts</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1806.05024.pdf">https://arxiv.org/pdf/1806.05024.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Lifelong Generative Modeling</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1705.09847.pdf">https://arxiv.org/pdf/1705.09847.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Adversarial Image Synthesis With Decision Tree Latent Controller</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.10603.pdf">https://arxiv.org/pdf/1805.10603.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Bidirectional Conditional Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.07461.pdf">https://arxiv.org/pdf/1711.07461.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Stacked Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1612.04357.pdf">https://arxiv.org/pdf/1612.04357.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Neural Photo Editing with Introspective Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1609.07093.pdf">https://arxiv.org/pdf/1609.07093.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Discriminative Domain Adaptation</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1702.05464.pdf">https://arxiv.org/pdf/1702.05464.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Controllable Generative Adversarial Network</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1708.00598.pdf">https://arxiv.org/pdf/1708.00598.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image
        Translation</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.09020.pdf">https://arxiv.org/pdf/1711.09020.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">A Style-Based Generator Architecture for Generative Adversarial Networks</td>
      <td
      style="text-align:left">StyleGAN allows users to control a wide range of style level features
        via AdaIn on each convolutional layer. This paper also introduce style-mixing,
        by assigning the parts of two latent code into high-level or low-level
        styles respectively. The author demonstrates that they are able to get
        high-level style from one image and low-level style from another via style-mixing</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1812.04948.pdf">https://arxiv.org/pdf/1812.04948.pdf</a>
          </td>
    </tr>
  </tbody>
</table>