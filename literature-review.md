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
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1505.07818.pdf">https://ar</a><a href="https://arxiv.org/pdf/1505.07818.pdf">xiv.org/pdf/1505.07818.pdf</a>
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
      <td style="text-align:left">Unsupervised Domain Adaptation by Backpropagation</td>
      <td style="text-align:left">This paper is about adding an adversarial component to force a classifier
        to ignore domain-specific information for successful domain adaptation.</td>
      <td
      style="text-align:left"><a href="http://proceedings.mlr.press/v37/ganin15.pdf">http://proceedings.mlr.press/v37/ganin15.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Deep Visual Domain Adaptation: A Survey</td>
      <td style="text-align:left">A deep survey on all types of domain adaptation categories. However, it
        has introduces more technical details than the general motivation for technique.
        For such a wide topic, such as domain adaption, the number of datasets
        used by the research community seems to be very limited.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1802.03601v4.pdf">https://arxiv.org/pdf/1802.03601v4.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">A Survey of Unsupervised Deep Domain Adaptation</td>
      <td style="text-align:left">This paper is a comprehensive survey of unsupervised domain adaption.
        Can read more to get leads to more papers.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1812.02849v2.pdf">https://arxiv.org/pdf/1812.02849v2.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Revisiting Batch Normalization For Practical Domain Adaptation</td>
      <td
      style="text-align:left">This paper observed that BN statistics in the two different datasets,
        Caltech-256 &amp; Bing can be separated almost perfectly with only a linear
        SVM. Hence, the authors attempted to modify BN layers to correct for the
        covariance shift and dataset bias with AdaBN. AdaBN worked beautifully
        for other datasets as well, and is independent from other domain adaptation
        techniques. So, AdaBN can be used on top of other forms of domain adaptation
        methods.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1603.04779.pdf">https://arxiv.org/pdf/1603.04779.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Open set domain adaptation</td>
      <td style="text-align:left">Interesting problem formulation where both source and larger dataset contains
        unknown labels. The solution seems to be based on alignment. Not sure.
        Dont understand technical implementation</td>
      <td style="text-align:left"><a href="https://bit.ly/2lF3axG">https://bit.ly/2lF3axG</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Open Set Domain Adaptation by Backpropagation</td>
      <td style="text-align:left">This is proof that adversarial training is applicable to a wider use-case
        compared to divergence/constrain-based technique. Constrain-based technique
        imposes assumption that are not valid in the case of open dataset.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1804.10427.pdf">https://arxiv.org/pdf/1804.10427.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Contrastive Adaptation Network for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left">Interesting idea to make use of class-awareness to improve the performance
        of domain adaptation. I would be interested to see how the network performance
        when the target domain do not contains same number of classes.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1901.00976.pdf">https://arxiv.org/pdf/1901.00976.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Domain Adaptation Meets Disentangled Representation Learning and Style
        Transfer</td>
      <td style="text-align:left">This paper uses the relation between style-transfer and domain-adaptation.
        For style-transfer, styles are domain-specific and contents are domain-generic.
        For domain-adaptation, only domain-generic information can be adapted to
        minimize negative transfer. The authors take advantage of this relationship
        to design a 3-in-1 system for disentangled representation learning, style-transfer,
        and domain-adaptation.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.09025.pdf">https://arxiv.org/pdf/1712.09025.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Simultaneous Deep Transfer Across Domains and Tasks</td>
      <td style="text-align:left">Don&apos;t understand</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1510.02192.pdf">https://arxiv.org/pdf/1510.02192.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Unsupervised Domain-Specific Deblurring via Disentangled Representations</td>
      <td
      style="text-align:left">This paper aims to disentangle blurring effect from an blurred image.
        This remains me of multi-model image-to-image translation where style is
        disentangled from content. In this case, blurring effect is disentangled
        instead of style. Can blurring be considered a style? What does not domain-shift
        of blurred image looked like? Does training on content information alone
        better for discriminatory task? Is the deblurring network adding information,
        or is it removing unimportant &apos;noise&apos;. From the papers, I infer
        that the network is unable to disentangle domain specific information.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1903.01594.pdf">https://arxiv.org/pdf/1903.01594.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss</td>
      <td
      style="text-align:left">This paper whiten features all features to standardize them to a sphere.
        Not sure about 2nd part on min-entropy consensus loss function. Interesting
        to see how whitening affect performance of vanilla network on imageNet
        dataset.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.03215.pdf">https://arxiv.org/pdf/1903.03215.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration</td>
      <td
      style="text-align:left">This paper introduce collaborative domain adaptation where both target
        and source domains are sparsely-labelled. I dont understand solution. Does
        not seems like using adversarial method.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1904.13179.pdf">https://arxiv.org/pdf/1904.13179.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Domain-Symmetric Networks for Adversarial Domain Adaptation</td>
      <td style="text-align:left">Don&apos;t understand</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1904.04663.pdf">https://arxiv.org/pdf/1904.04663.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Universal Domain Adaptation</td>
      <td style="text-align:left">
        <p>This author published 3 papers in 2019. He is definitely a raising star
          in domain adaptation. His universal domain adaptation architecture is amazing.
          It relies on assumption that I believe to be very general.</p>
        <p></p>
        <p>E c&apos; p(d&apos;) &gt; E c p(d&apos;) &gt; E c q(d&apos;) &gt; E c&apos;
          q(d)</p>
        <p>E c&apos; p(H(y)) &lt; E c p(H(y)) &lt; E c q(H(y)) &lt; E c&apos; q(H
          (y)</p>
        <p></p>
        <p>In additional, the author did not use the adversarial component as it
          was trained to be fooled. Instead, he used a domain similarity metric constructed
          from the above assumption to determined out-of-class examples. It reminds
          me of a hypothesis that implies that a collaborative task in GANs help
          to stabilize and improve the overall performance of the network.</p>
        <p></p>
        <p>High-quality paper.</p>
      </td>
      <td style="text-align:left"><a href="https://youkaichao.github.io/files/cvpr2019/1628.pdf">https://youkaichao.github.io/files/cvpr2019/1628.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Learning to Transfer Examples for Partial Domain Adaptation</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://youkaichao.github.io/files/cvpr2019/1855.pdf">https://youkaichao.github.io/files/cvpr2019/1855.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://youkaichao.github.io/files/icml2019/923.pdf">https://youkaichao.github.io/files/icml2019/923.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding</td>
      <td
      style="text-align:left">
        <p>This paper claims to achieve domain generalization using d-SNE, and also
          outperforms unsupervised techniques that have access to all samples in
          target domain. If these claims are true, this paper will be a very interesting
          paper.</p>
        <p>Domain adaptation strategy: 1) Domain transformation from target to source,
          2) Latent-space transformation to learn domain-invariant features which
          maps both source and target domain into a common latent space.</p>
        <p>With only 3 samples per class, this paper managed to outperform other
          techniques that uses all of the target domain. The logic behind this code
          is to minimize the inverse-probability for class matching, with probability
          calculated using distance between the source and target samples. The main
          assumption is that we want intra-class probability to be maximized, and
          inter-class probability to be minimized.</p>
        <p>Not sure, if I understand the exact implementation, but I think I understand
          the big picture. I may need to implement to have know what I messed out.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.12775.pdf">https://arxiv.org/pdf/1905.12775.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Domain Generalization by Solving Jigsaw Puzzles</td>
      <td style="text-align:left">A uninteresting paper with little to no theory of its own. The main idea
        is to use the well-known self-supervising techniques for domain generalization.
        No benchmark for office-31 dataset which is the standard test. Did not
        read thoroughly, so I might be wrong.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.06864.pdf">https://arxiv.org/pdf/1903.06864.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">CrDoCo: Pixel-Level Domain Transfer With Cross-Domain Consistency</td>
      <td
      style="text-align:left">
        <p>Feature-level vs Pixel-level? I only know feature level domain adaptation.
          This paper let me understand that CycleGANs is also part of domain-adaptation
          that uses pixel-to-pixel translation.</p>
        <p>This literature is closer to CycleGAN which the author coined as &quot;dense
          prediction&quot; i.e. pixel-level prediction. It assumes that two the local
          task of source and target domain yields the same results.</p>
        </td>
        <td style="text-align:left"><a href="https://bit.ly/2lGVfzK">https://bit.ly/2lGVfzK</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Generalizable Person Re-identification by Domain-Invariant Mapping Network</td>
      <td
      style="text-align:left">Domain generalization without target samples, or re-training. It seems
        too good to be true. Although it is close to domain adaption, I am dont
        understand much. Come back to finish this paper soon.</td>
        <td style="text-align:left"><a href="http://www.eecs.qmul.ac.uk/~js327/Doc/Publication/2019/cvpr2019_dimn.pdf">http://www.eecs.qmul.ac.uk/~js327/Doc/Publication/2019/cvpr2019_dimn.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Progressive Feature Alignment for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left">
        <p>This is the first paper I have read that implies that pseudo-labels are
          being used in the field of domain adaptation. According to the author,
          many papers assumes that the pseudo labels assigned to each samples to
          guide the network during training time, and the performance of these network
          are held back by these falsely-labeled samples.</p>
        <p>I don&apos;t understand the technical jargon that are the main contribution
          of the paper.</p>
        <p>Hard samples are those that are dissimilar to source domain that are far
          away. False-easy samples lies in the decision boundary of the wrong class,
          resulting in false labels despite high-confidence. The so-called &quot;Easy-to-Hard
          Transfer Strategy&quot; is somehow going to solve this. I havent read finish
          yet.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1811.08585.pdf">https://arxiv.org/pdf/1811.08585.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification</td>
      <td
      style="text-align:left">This paper is about domain generalization, but I do not see any component
        that domain generlization capabilites. I am unfamilar with ReID technology.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1904.01990.pdf">https://arxiv.org/pdf/1904.01990.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Attending to Discriminative Certainty for Domain Adaptation</td>
      <td style="text-align:left">This paper is used to identify areas of an image that can be adapted,
        instead of using the entire image. Maybe this approach is a way for unsupervised
        attention i.e. no pixel level labeling. The results are impressive as one
        of the highest accuracy model. Unlike universal domain adaptation, this
        model attempt to find areas of a images that can be adapted. Maybe can
        combine with universal domain adaptation</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1906.03502.pdf">https://arxiv.org/pdf/1906.03502.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation</td>
      <td
      style="text-align:left">Looks interesting. I dont understand.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1705.01314.pdf">https://arxiv.org/pdf/1705.01314.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Importance Weighted Adversarial Nets for Partial Domain Adaptation</td>
      <td
      style="text-align:left">Accuracy is lower than universal domain adaptation which is more general
        and provides more explanation.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1803.09210.pdf">https://arxiv.org/pdf/1803.09210.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Efficient parametrization of multi-domain deep neural networks</td>
      <td
      style="text-align:left">Interesting idea of a 3-in-1 architecture for multi-tasking, multi-domain
        &amp; lifelong learning. I was expecting some explanation, but instead
        got only experimental results. In a nutshell, this paper is about how to
        most efficiently share parameters for the above three objectives.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1803.10082.pdf">https://arxiv.org/pdf/1803.10082.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Unsupervised Domain Adaptation with Similarity Learning</td>
      <td style="text-align:left">Replacing traditional softmax layer with similarity comparison with each
        class prototype. Shorting distance with class-avg is the predicted class.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1711.08995.pdf">https://arxiv.org/pdf/1711.08995.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Camera Style Adaptation for Person Re-identification</td>
      <td style="text-align:left">Style-transfer using &quot;Camera-Style&quot; for image augmentation for
        persion ReID.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.10295.pdf">https://arxiv.org/pdf/1711.10295.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Image to Image Translation for Domain Adaptation</td>
      <td style="text-align:left">
        <p>This paper is about image-to-image translation using adversarial encoder.
          The network have 3 tasks. Encoding &amp; Decoding in source domain, Encoding
          &amp; Decoding in target domain, Encoding &amp; Decoding across source
          &amp; target domain.</p>
        <p>Seems to me like multi-tasking. However, multi-tasking sacrifice accuracy
          for generalization, this method achieved avg accuracy of 70~% for office-31
          data set. Much lower than state-of-the-art of 89-90+%.</p>
        <p>Interesting architecture I guess.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.00479.pdf">https://arxiv.org/pdf/1712.00479.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Duplex Generative Adversarial Network for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left">
        <p>Interesting architecture design. This approach uses a generative design
          instead of discriminative design. Ds is trained using actual labels, while
          Dt is trained using pseudo labels. Hence, accuracy of pseudo labels will
          affect performance greatly. &quot;Progressive Feature Alignment for Unsupervised
          Domain Adaptation&quot; paper have some technique to address wrong pseudo-labels.</p>
        <p>The rationale for this design is that the generator will need domain-invariant
          encoding to fool both Ds &amp; Dt.</p>
        <p>Accuracy on Office-31 seems to beat SOTA for previous years, but it is
          far behind same year SOTA.</p>
        </td>
        <td style="text-align:left"><a href="https://bit.ly/2klnPGY">https://bit.ly/2klnPGY</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Unified Deep Supervised Domain Adaptation and Generalization</td>
      <td style="text-align:left">
        <p>Similar to &quot;d-SNE: Domain Adaptation using Stochastic Neighborhood
          Embedding&quot; where only few labelled examples from target data set is
          needed. It seems to me that this approach is semi-supervised, instead of
          supervised. I guess it can be both as the author mentioned that &quot;We
          aim at handling cases where there is only one target labeled sample, and
          there can even be some classes with no target samples at all&quot;. In
          summary, this paper tries to use limited labeled images as as a &quot;anchor
          point&quot; (my own words, no the author&apos;s) to calculate the known
          disparity between the target and source, and using that information to
          extrapolate to other other unlabeled target images.</p>
        <p>Not 100% sure of the technical implementation.
          <br />
        </p>
        <p>f = h.g, g: X -&gt; Z, h: Z-&gt; Y</p>
        <p>Therefore, features invariant should be at g. Authors propose to use shared
          weighs in g.</p>
        <p>Samples of the same class are mapped together in SDA, unlike UDA. This
          addresses one of my concerns where class-alignment between domains are
          not enforced. The phenomenon is coined as &quot;semantic alignment&quot;</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1709.10190.pdf">https://arxiv.org/pdf/1709.10190.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">AutoDIAL: Automatic DomaIn Alignment Layers</td>
      <td style="text-align:left">I am interesting in implementing this. However, I cannot understand the
        implementation details. I need to learn basic back-propagation. Appendix
        of paper is important.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1704.08082.pdf">https://arxiv.org/pdf/1704.08082.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Domain-adaptive deep network compression</td>
      <td style="text-align:left">Paper about pruning of neural network that applies post-domain adapted
        activation statistics for more efficient network compression.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1709.01041.pdf">https://arxiv.org/pdf/1709.01041.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">PUnDA: Probabilistic Unsupervised Domain Adaptation for Knowledge Transfer
        Across Visual Categories</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://bit.ly/2k77OUK">https://bit.ly/2k77OUK</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Associative Domain Adaptation</td>
      <td style="text-align:left">The idea of this paper is similar to &quot;Contrastive Adaptation Network
        for Unsupervised Domain Adaptation&quot;, where the labels from source
        dataset is used to enforce an constraint on the features on the target
        dataset. The difference between the two papers are contrastive vs associative.
        I believe this idea assumes that have a implicit assumption that the number
        of classes across domain remains the constant. Not relevant for me.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1708.00938.pdf">https://arxiv.org/pdf/1708.00938.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Fine-grained Recognition in the Wild: A Multi-Task Domain Adaptation Approach</td>
      <td
      style="text-align:left">Fine-grain domain adaptation seems to be a new field with this model claiming
        to have achieved 19.1% accuracy (which is low). Main challenges of fine-grain
        DA is the limited availability of dataset. This method make use of a attributes
        to improve object classification (a term that I am unfamiliar with in the
        context of deep learning)</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1709.02476.pdf">https://arxiv.org/pdf/1709.02476.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner</td>
      <td
      style="text-align:left">Not in scope, but looks interesting. The model is using paired images-words
        in source domain to train unpaired images-words in target domain, using
        adversarial methods. It seems to me like its more of a task transfer, as
        the style of wording in the target domain is different. Perhaps its a mixture
        of domain and task transfer.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1705.00930.pdf">https://arxiv.org/pdf/1705.00930.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Maximum Classifier Discrepancy for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://bit.ly/2GwJUuK">https://bit.ly/2GwJUuK</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Generate To Adapt: Aligning Domains using Generative Adversarial Networks</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1704.01705.pdf">https://arxiv.org/pdf/1704.01705.pdf</a>
        </td>
    </tr>
  </tbody>
</table>### Fourier Neural Network

| Name | Summary | File |
| :--- | :--- | :--- |
| Single-Image Depth Estimation Based on Fourier Domain Analysis |  | [https://bit.ly/2lH0kIk](https://bit.ly/2lH0kIk) |

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
| Unsupervised Visual Representation Learning by Context Prediction | This paper propose a interesting way for representation learning by training the network to predict the relative position of a source image to the target image. Experiment shows that once the authors removed so-called "trival" solutions that rely on low-level features, the network is able to start learning more advanced features. The author claims that theirs in the first example of unsupervised pre-training on a larger dataset could led to performance boost in a smaller dataset. | [https://arxiv.org/pdf/1505.05192.pdf](https://arxiv.org/pdf/1505.05192.pdf) |
| Unsupervised Representation Learning By Predicting Image Rotation | This paper propose a simple and, yet powerful approach that allows the learning of high-level semantic features, by predicting the rotational orientation. Compared to similar approaches, there is no need to additional pre-processing to avoid "trivial" solutions | [https://arxiv.org/pdf/1803.07728.pdf](https://arxiv.org/pdf/1803.07728.pdf) |
| Colorful Image Colorization | The main focus of this paper is to use deep learning for multi-modal re-colorisation of images. On the side, the author endeavor into a cross-channel encoder and using the features learnt for classification task. Thus fur, I observed that self-supervised techniques can be broadly grouped into three styles: 1\) hiding information away from learner,  2\) feature learning before & after data augmentation, and 3\) self-labeling  | [https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf) |
| Stand-Alone Self-Attention in Vision Models |  |  |
|  |  |  |

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
        and is worthy of further investigation.</td>
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
        This may be useful if flow-based generator becomes a research focus.</td>
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
      <td style="text-align:left">This paper combine the solution for robust-noise classifier with cGANs
        &amp; AcGANS. Not interested.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1811.11165.pdf">https://arxiv.org/pdf/1811.11165.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarially Learned Inference</td>
      <td style="text-align:left">BiGans-twin</td>
      <td style="text-align:left"></td>
    </tr>
    <tr>
      <td style="text-align:left">Large Scale Adversarial Representation Learning</td>
      <td style="text-align:left">BigBiGans</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.02544.pdf">https://arxiv.org/pdf/1907.02544.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Collaborative Sampling in Generative Adversarial</td>
      <td style="text-align:left">This paper propose a way to reuse GAN&apos;s discriminator instead of
        simple discarding it. The authors augment the generator&apos;s output with
        feedback from the discriminator. Interesting idea.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1902.00813.pdf">https://arxiv.org/pdf/1902.00813.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional image synthesis with auxiliary classifier GANs</td>
      <td style="text-align:left">The first paper on cGANs &amp; AcGANs. Apart from novel conditioning,
        this paper introduces discriminatory as a metric to approximate effective
        resolution, citing that low resolution images are less discriminate. Paper
        also suggest MS-SSIM for measuring diversity, but from my understanding
        it is commonly used.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1610.09585.pdf">https://arxiv.org/pdf/1610.09585.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Compression</td>
      <td style="text-align:left">This paper presents a GANs-based compression technique for encoding an
        image via a neural network. I believe that much of the latent information
        has been encoded into the network itself, such that the actual compression
        factor has been massively increased. I believe that this technique is effectively
        representation learning with image/video compression as a use-case. It
        will be interesting to see reconstruction accuracy for out-of-class images.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1703.01467.pdf">https://arxiv.org/pdf/1703.01467.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">It Takes (Only) Two: Adversarial Generator-Encoder Networks</td>
      <td style="text-align:left">Encoder-Decoder GANs papers with lots of mathematical proof, and some
        experimental results. Only two components G &amp; E are needed, compared
        to 3 in BiGANs. The idea is interesting, and is worth further investigation.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1704.02304.pdf">https://arxiv.org/pdf/1704.02304.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Inverting the Generator of a Generative Adversarial Network</td>
      <td style="text-align:left">This paper is an early work on the invertibility of generator, which have
        now been achieve by BiGANs.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1611.05644.pdf">https://arxiv.org/pdf/1611.05644.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Feature Augmentation for Unsupervised Domain Adaptation</td>
      <td
      style="text-align:left">This paper uses GANs to augment feature map of target domain to have similar
        encoding as the source domain.</td>
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
      <td style="text-align:left">Text conditional GANs. Introduction contains many papers using various
        types of conditioning.</td>
      <td style="text-align:left"><a href="https://bit.ly/30iJtuQ">https://bit.ly/30iJtuQ</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Structured Generative Adversarial Networks</td>
      <td style="text-align:left">This paper addresses enables controllability of GANs without a fully-labeled
        dataset. I do not understand the technical details yet. I believe that
        a deep dive needed.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.00889.pdf">https://arxiv.org/pdf/1711.00889.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional Image-to-Image Translation</td>
      <td style="text-align:left">This paper is seems to be very similar to style-transfer. Moreover, the
        architecture contains 4 GANs which would be incredibly hard to train. I
        believe that there must be way to image-conditioning simpler</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.00251.pdf">https://arxiv.org/pdf/1805.00251.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Ensembles of Generative Adversarial Networks</td>
      <td style="text-align:left">This paper contains evaluation of GANs ensembles, namely; simple ensemble,
        GANs, ensemble at each iteration GANs &amp; cascade GANs. Authors indicate
        that seGANs &amp; cGANs can be combined, and that are there isn&apos;t
        a good way to evaluate GAN ensemble.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1612.00991.pdf">https://arxiv.org/pdf/1612.00991.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Denoising Adversarial Autoencoders</td>
      <td style="text-align:left">Applying a denoising criterion improve encoding for classification. Dont
        understand.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1703.01220.pdf">https://arxiv.org/pdf/1703.01220.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">How Generative Adversarial Networks and Their Variants Work: An Overview</td>
      <td
      style="text-align:left">Survey of GANs. Good to read if, I wish to understand technical details.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1711.05914.pdf">https://arxiv.org/pdf/1711.05914.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Self-Supervised Feature Learning by Learning to Spot Artifacts</td>
      <td
      style="text-align:left">This paper introduce a self-supervising technique to that in-paints realistic
        artifacts that are locally unnoticeable, but globally incorrect. The authors
        shows that their method achieved SOTA for unsupervised representation learning.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1806.05024.pdf">https://arxiv.org/pdf/1806.05024.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Lifelong Generative Modeling</td>
      <td style="text-align:left">This paper introduces a way to mitigate catastrophic forgetting when learning
        new tasks sequentially. The student model becomes the teacher at every
        new task, hence preserving the knowledge from previously learnt task.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1705.09847.pdf">https://arxiv.org/pdf/1705.09847.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Adversarial Image Synthesis With Decision Tree Latent Controller</td>
      <td
      style="text-align:left">Interesting idea, but paper is not presented in an easy to read manner.
        The core idea is to impose a hierarchical order in the latent code, and
        a regularization component to force the network to learn disentangled representation
        in a a weakly supervised manner. Do not understand the technical details.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1805.10603.pdf">https://arxiv.org/pdf/1805.10603.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Invertible Conditional GANs for image editing</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1611.06355.pdf">https://arxiv.org/pdf/1611.06355.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Bidirectional Conditional Generative Adversarial Networks</td>
      <td style="text-align:left">This paper aims to apply BiGANs to cGANs, hence the name. The problem
        statement is that traditional cGANs is unable to disentangle instrinic
        code and extrinsic code, which the author claims to be the &quot;desired
        properties of cGANs&quot; Naively combining BiGANs and cGANS will not work.
        Hence, the solution is to c&apos; supervision. Honestly, I am not impressed
        because it seems to be a incremental improvement.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.07461.pdf">https://arxiv.org/pdf/1711.07461.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Stacked Generative Adversarial Networks</td>
      <td style="text-align:left">This paper attempt to train a generator by leveraging hierarchical representation
        from discriminators. Instead of a single realism discriminator like a typical
        GANs, sGANs divide them into several smaller stacked GANs, each with their
        own representation discriminator. Experiment conducted by the authors show
        that their architecture outperforms GANs w/o stacking. The authors omit
        a lot of technical implementation details, I would have to dive into the
        source code to figure out exact logic.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1612.04357.pdf">https://arxiv.org/pdf/1612.04357.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Neural Photo Editing with Introspective Adversarial Networks</td>
      <td style="text-align:left">This paper is about a tool to make semantic changes to a image, instead
        of making pixel-wise modification to a photo. The tool is a hybird between
        VAE and GANs, to &quot;improve the capacity of the latent space without
        increasing its dimension-ality&quot; as &quot;features learned by a discriminatively
        trained network tend to be more expressive&quot;</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1609.07093.pdf">https://arxiv.org/pdf/1609.07093.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Discriminative Domain Adaptation</td>
      <td style="text-align:left">This paper proposed to force images in the target domain to mimic encodings
        from the source domain. Experiments shows that for numbers dataset, their
        architecture achieved SOTA. However for the more complex NYUD dataset,
        no comparison with existing methods was conducted. The flaw in the methodology
        is that having similar encodings does not ensure that class information
        is preserved. For example, the target network can encodes image of number
        ONE to the encoding of number THREE, and the discriminator will not be
        able to differentiate apart.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1702.05464.pdf">https://arxiv.org/pdf/1702.05464.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Controllable Generative Adversarial Network</td>
      <td style="text-align:left">ControlGANs seems very much like AcGANs, if input labels is 0 and 1. This
        paper explores interpolation of labels between 0 - 1 and extrapolation,
        when labels &lt;0 and &gt;1. Results from the author shows, -1x labels
        produced negative examples and 2x labels produces exaggerated examples.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1708.00598.pdf">https://arxiv.org/pdf/1708.00598.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image
        Translation</td>
      <td style="text-align:left">This paper propose a image+domain conditioning for image-to-image translation.
        Top-down approach paper with great utility and technically simple. Incremental
        paper.</td>
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
    <tr>
      <td style="text-align:left">Improved Techniques for Training GANs</td>
      <td style="text-align:left">...</td>
      <td style="text-align:left"><a href="https://papers.nips.cc/paper/6125-improved-techniques-for-training-gans">https://papers.nips.cc/paper/6125-improved-techniques-for-training-gans</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">DS3L: Deep Self-Semi-Supervised Learning for Image Recognition</td>
      <td
      style="text-align:left">...</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.13305.pdf">https://arxiv.org/pdf/1905.13305.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">On Self Modulation for Generative Adversarial Networks</td>
      <td style="text-align:left">Self-Modulation Block looks very similar to AdaIn. So far, there is not
        a hint of any explaination on why self-modulation works. It is merely an
        idea that &quot;works&quot;.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1810.01365.pdf">https://arxiv.org/pdf/1810.01365.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Large Scale GAN Training for High Fidelity Natural Image Synthesis</td>
      <td
      style="text-align:left">This is the BiGAN paper, which is a good collection of techniques used
        to train GANs on a large scale.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1809.11096.pdf">https://arxiv.org/pdf/1809.11096.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Deep Generative Image Models using a Laplacian Pyramid of Adversarial
        Networks</td>
      <td style="text-align:left">Interesting idea. However, ProGAN implemented something similar and more
        elegant solution, a few years since this.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1506.05751.pdf">https://arxiv.org/pdf/1506.05751.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">cGANs with Projection Discriminator</td>
      <td style="text-align:left">This paper introduces a projection-based conditioning derived from mathematical
        equations. Not my style, and not much improvement can be made. Interesting
        way of conditioning though.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1802.05637.pdf">https://arxiv.org/pdf/1802.05637.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Transferring GANs: generating images from limited data</td>
      <td style="text-align:left">This paper is an empirical study of transfer learning of GANs trained
        on large dataset like ImageNet, to dataset which are typically too small
        to be trained reasonably. Interesting study, but the results are mostly
        expected i.e. transfering both generator and discriminate yields the best
        results.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.01677.pdf">https://arxiv.org/pdf/1805.01677.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">A Learned Representation from Artistic Style</td>
      <td style="text-align:left">The paper was cited by AdaIn paper. AdaIn paper generalised affine transformation
        in batch normalization layer for arbitrary style.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1610.07629.pdf">https://arxiv.org/pdf/1610.07629.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Collaborative Sampling in Generative Adversarial Networks</td>
      <td style="text-align:left">This paper make use of the discriminator&apos;s feedback to improve the
        generator&apos;s output, instead of discarding it. The downside is that
        every image generated classified as &apos;fake&apos; would need to be trained
        in the refinement layer.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1902.00813.pdf#cite.turner_metropolis-hastings_2018">https://arxiv.org/pdf/1902.00813.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Discriminator Rejection Sampling</td>
      <td style="text-align:left">This paper is cited by &apos;Collaborative Sampling in GANs&apos;. The
        idea resolves around rejecting generated samples that deems as fake by
        the discriminator. The approach improves the quality of image, at the cost
        of image diversity.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1810.06758.pdf">https://arxiv.org/pdf/1810.06758.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization</td>
      <td
      style="text-align:left">This paper enables real-time &amp; arbitary style transfer using AdaIn.
        The core idea is that the variance and means is value contains style information.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1703.06868.pdf">https://arxiv.org/pdf/1703.06868.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">SGAN: An Alternative Training of Generative Adversarial Networks</td>
      <td
      style="text-align:left">I am not convinced of this method proposed by this paper. Results shown
        in this paper is not impressive. There is no theory, and result is not
        good (my opinion).</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.02330.pdf">https://arxiv.org/pdf/1712.02330.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">CartoonGAN: Generative Adversarial Networks for Photo Cartoonization</td>
      <td
      style="text-align:left">CycleGAN is not good at preserving edges. So, authors artificial remove
        edges from scene to allow the network to include. They initialize GANs
        to preserve only content before training. Seems like a task-specific improvement.
        But, I believe that some ideas can be reused.</td>
        <td style="text-align:left"><a href="https://bit.ly/2lFooeI">https://bit.ly/2lFooeI</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Multi-Agent Diverse Generative Adversarial Networks</td>
      <td style="text-align:left">
        <p>This paper proposed MAD-GANS. Multi-generators &amp; Single Discriminator
          and enforce diversity between generators, which directed each generator
          towards a different modality. Hence, addressing modal collapse. To enforce
          dissimilarity between the generators, the loss function of the discriminator
          is modified, such that it have to guess the exact generator which produced
          the image. I believe this will also help GANs to converge by making discrimination
          harder, hence preventing D from being too strong.</p>
        <p>Weakness of this approach is that, the number of modes must match the
          number of generators for best results.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1704.02906.pdf">https://arxiv.org/pdf/1704.02906.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Single Image Dehazing via Conditional Generative Adversarial Network</td>
      <td
      style="text-align:left">
        <p>Great top-down paper. Starting with an objective to de-haze images. From
          applying existing techniques to expose flaws, to proposing task-specific
          solutions to bypass the limitations, until the ending model produces good
          result. The paper also discuss limitation of approach i.e light hazing/
          night hazing.</p>
        <p>The authors made extensive use of synthetic data generated using the hazing
          model and image guided filtering method.</p>
        <p>I believe that the ability to generate synthetic data is key for producing
          papers like this one.</p>
        </td>
        <td style="text-align:left"><a href="https://bit.ly/2kmBdur">https://bit.ly/2kmBdur</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Face Aging with Identity-Preserved Conditional Generative Adversarial
        Networks</td>
      <td style="text-align:left">Similar to &quot;Single Image Dehazing via Conditional Generative Adversarial
        Network&quot;, starts with vanilla GANs. Then, moved to LSGANs for better
        stablility and better quality iimages. Then, introduces identity-preserving
        module to maintain high-level content features at earlier layers, while
        allowing lower-level &apos;age&apos; related features to vary. Lastly,
        age loss is introduced to allow correct aged images to be generated.
        <br
        />Start with generic case and incrementally solve problems.
        <br />Interesting application: Aged-faced detection.</td>
      <td style="text-align:left"><a href="https://bit.ly/2kxODn9">https://bit.ly/2kxODn9</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Eye In-Painting with Exemplar Generative Adversarial Networks</td>
      <td
      style="text-align:left">
        <p>This paper is about in-painting eyes. The challenges faced is the uncanny
          valley, where humans can easily pickup on small errors. Secondly, normal
          GANs cannot augment the eyes without disturbing the rest of the images.
          Thirdly, the in-painted eyes need to blend-in with the rest of the image,
          while preserving person-specific features.</p>
        <p>Authors proposed two methods: 1) Reference image, and 2) Code-based in-painting.</p>
        <p>Both methods use an exemplar image as reference. However, the code-based
          approach first encodes the reference image before conditioning the generator.
          The discriminator differentiate using the code, instead of the whole image.</p>
        <p>
          <br />I believe exemplar concept is interesting and can be utilized for other
          applications.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.03999.pdf">https://arxiv.org/pdf/1712.03999.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Look, Imagine and Match:Improving Textual-Visual Cross-Modal Retrieval
        with Generative Models</td>
      <td style="text-align:left">Paper about cross-modal image-caption retrieval</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.06420.pdf">https://arxiv.org/pdf/1711.06420.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Logo Synthesis and Manipulation with Clustered Generative Adversarial
        Networks</td>
      <td style="text-align:left">
        <p>This paper introduces Layer-conditioning GANs where the pseudo label is
          obtain autoencoder clustering/resnet classifier clustering, which are conditioned
          at all layers. From previous papers, conditioning at each layers will influence
          the higher-to-lower level details, depending on the depth of layers conditioned.
          Also authors used gaussian blurring to help stablised training of GANs.</p>
        <p>ACGANs have many qualitative advantages in terms of image diversity, but
          LCGANs allows &quot;smooth&quot; transition between logos, which is a beneficial
          property for the intended use-case.
          <br />The authors also managed to obtained &quot;sharping&quot; vector by subtracting
          z-encoding of blur images from sharp images, which can be added to blur
          images to sharpen them (No ideas how this works, but maybe worth exploring).</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.04407.pdf">https://arxiv.org/pdf/1712.04407.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Adversarial Learning Towards Fast Weakly Supervised Detection</td>
      <td
      style="text-align:left">This paper propose using GANs to speed up existing weakly supervised object
        detection. Weakly supervised detection is outside my scope, but there may
        be some interesting ideas. I don&apos;t really understand this.</td>
        <td
        style="text-align:left"><a href="https://bit.ly/2lyTjtp">https://bit.ly/2lyTjtp</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">DA-GAN: Instance-level Image Translation by Deep Attention Generative
        Adversarial Networks (with Supplementary Materials)</td>
      <td style="text-align:left">
        <p>This paper introduces attention mechanism to achieve better results with
          lesser semantic and structural artifacts. Another benefit is the domain
          adaptation capability. My opinion is that the paper is poorly written and
          many components are not well explained. Although results are good, the
          reasoning and motivation of each new component is not clear.</p>
        <p>For example, what is instance-level? Why use Deep Attention Encoder? F
          = DAE * G is before explaining what it is. Great results, Terrible writing.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1802.06454.pdf">https://arxiv.org/pdf/1802.06454.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Image In-painting with Contextual Attention</td>
      <td style="text-align:left">
        <p>This paper improves on previous works where image in-painting copies surrounding
          textures into the empty space. The approach works only if the scene in
          general consists of repeating patterns, and works well for in-painting
          in the background. The authors improves existing work by enabling in-painting
          of content structure such as faces and objects.</p>
        <p>In the lit review section, the authors argues that CNN have trouble modelling
          long-range correlation, as two pixels that 64 pixels apart requires at
          large number of 3x3 conv such that they are within the same receptive field.</p>
        <p>The weights of pixels with closest known pixel is higher, using concept
          from reinforced learning for delayed gratification. Also authors use coarse
          in-painting followed by fine in-painting to achieve better novel reconstruction
          for faces &amp; objects.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1801.07892.pdf">https://arxiv.org/pdf/1801.07892.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Unsupervised Deep Generative Adversarial Hashing Network</td>
      <td style="text-align:left">
        <p>Unsupervised GANs hashing seems to be quite similar to unsupervised representation
          learning. However, hashing looks like to have additional use cases, such
          as image retrieval.</p>
        <p>Largely, the architecture is very similar to ACGANs w/o categorical conditioning.
          The main difference is the output hash, several ideal properties of a &quot;good
          hash&quot; is enforced in the loss function, such as minimum entropy bits,
          uniform frequency bits, consistent bits &amp; independent bits. At least,
          that is my understanding.</p>
        <p>Perhaps some kind of hierarchical hashing is possible.</p>
      </td>
      <td style="text-align:left"><a href="https://bit.ly/2m1Szgy">https://bit.ly/2m1Szgy</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Image Blind Denoising With Generative Adversarial Network BasedNoise Modeling</td>
      <td
      style="text-align:left">
        <p>Interesting paper that uses GAN for noising modelling. Then the trained
          GANs is used to add noise to clean images, which will be used as paired
          examples for training the denoiser. My thoughts is that the so called &quot;clean
          images&quot; are critical for this architecture to work. Depending on the
          standards set for &quot;clean&quot;, results may vary.</p>
        <p>I believe that this work diverge from Noise2Noise. (Noise2Noise is a Good
          paper)</p>
        </td>
        <td style="text-align:left"><a href="https://bit.ly/2m1JuEr">https://bit.ly/2m1JuEr</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Attentive Generative Adversarial Network for Raindrop Removal from a Single
        Image</td>
      <td style="text-align:left">This paper is able using attention models, as a prior to aid in the reconstruction
        of images that have been corrupted by raindrops. The catch is that the
        segmentation areas to focus on, is learnt by network itself. Maybe I can
        call this self-prior-ing? Perhaps its a subset of self-supervised learning.
        Either way, I believe there must be other application of this interesting
        technique, like &quot;Generative Image In-painting with Contextual Attention&quot;</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1711.10098.pdf">https://arxiv.org/pdf/1711.10098.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative
        Adversarial Networks</td>
      <td style="text-align:left">
        <p>Novel dataset, proposed architecture and refining approach.</p>
        <p>Time-lapse dataset, 3d U-Net GANs, and Refine-Net</p>
        <p>Skip-connections are good for perserving content, but not ideal for preserving
          motion. Maximize distance of image from 2nd stage to 1st stage and minimize
          distance of 2nd stage with ground truth.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1709.07592.pdf">https://arxiv.org/pdf/1709.07592.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Social GAN: Socially Acceptable Trajectories with Generative Adversarial
        Networks</td>
      <td style="text-align:left">
        <p>This paper aims to narrow-down the possible trajectory that a person is
          able to take by enforcing social-acceptability. Many paths are physically
          possible, but not all are socially acceptable.</p>
        <p>Current techniques use loss function that would implicitly result in a
          &apos;mean&apos; trajectory. This paper aims to solve that problem by using
          adversarial loss, which will learn social acceptability instead (by learning
          how people interacts with each other to move in a crowd).</p>
        <p>The network encodes all movement then encodes the historical movement
          to each person. Then generator will take encoding of every person to generate
          motion based on past locations. This ensure that motion prediction is based
          on interactions of all people, instead of just the surrounding neighbors.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1803.10892.pdf">https://arxiv.org/pdf/1803.10892.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Stacked Conditional Generative Adversarial Networks for Jointly Learning
        Shadow Detection and Shadow Removal</td>
      <td style="text-align:left">This paper introduces a multi-tasking framework to both detect and remove
        shadows. The approach is to 1st detect shadows, then introduces it as a
        prior to aid in shadow removal. This literature make use of shadow map
        in adversarial training even when supervised learning is possible. I believe
        something like &quot;Attentive Generative Adversarial Network for Raindrop
        Removal from a Single Image&quot; is possible.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1712.02478.pdf">https://arxiv.org/pdf/1712.02478.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Global versus Localized Generative Adversarial Nets</td>
      <td style="text-align:left">This paper propose to use local coordinate system for GANs. I believe
        that this idea is very general. It maybe closer to representation learning
        than to GANs. Very interesting ideas indeed. Perturbating the encoding
        locally, insteading of sampling of a global coordinate. I think Iocalized
        GAN must be cGAN (I may be wrong).</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1711.06020.pdf">https://arxiv.org/pdf/1711.06020.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Disentangled Representation Learning GAN for Pose-Invariant Face Recognition</td>
      <td
      style="text-align:left">This paper seek to learn pose-invariant features using DR-GANs. DRGANs
        can not only classify person, but also to general alternate face angles.
        I believe that key difficulty in the task is getting the dataset. The idea
        itself is almost trival. It is just ACGANs w/ autoencoder.</td>
        <td style="text-align:left"><a href="https://bit.ly/2jYjGZ4">https://bit.ly/2jYjGZ4</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Semi Supervised Semantic Segmentation Using Generative Adversarial Network</td>
      <td
      style="text-align:left">This paper uses a multi-tasking framework to achieve good segmentation
        results. The main task to learn correct dense pixel class i.e. segmentation
        from labelled dataset. The second task is to generate realistic images.
        The idea is that the features learnt from trying to generating realistic
        images is related to segmentation task. Hence, it serves as a regularization
        factor.</td>
        <td style="text-align:left"><a href="https://bit.ly/2kzaPNT">https://bit.ly/2kzaPNT</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Shadow Detection with Conditional Generative Adversarial Networks</td>
      <td
      style="text-align:left">According to this paper, detecting shadows with varying intensity used
        to be require multiple trained network. In this paper, the authors solved
        this problem, by introducing a sensitivity input for adjusting the threshold
        for the shadow map.</td>
        <td style="text-align:left"><a href="https://bit.ly/2lZbd8C">https://bit.ly/2lZbd8C</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Multimodal Unsupervised Image-to-Image Translation</td>
      <td style="text-align:left">In this paper, the authors aims to achieve multimodal image-to-image translation
        as the title suggest. They do this by assuming that the style and content
        can be disentangled. Once disentangled, each content can be translated
        to various styles. The architecture uses AdaIn for style-coding and residual-blocks
        to hold content information. This architecture is very commonly used, to
        encode styles into content. Authors also added some technical terms into
        loss function which I do not understand yet. Great results, but I believe
        it is because the network is trained on target domain and the content image
        and target image have similar structure.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1812.02849v2.pdf">https://arxiv.org/pdf/1812.02849v2.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
        Networks</td>
      <td style="text-align:left">This is the CycleGANs paper. They uses cycle-consistency constraint to
        stabilize unpaired image--to-image translation. I believe all results are
        curated since, I see in other papers that CycleGANs give poor results.
        All other papers are most likely-curated also, need to be careful.</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1703.10593.pdf">https://arxiv.org/pdf/1703.10593.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Modeling Tabular data using Conditional GAN</td>
      <td style="text-align:left">This paper is about modelling tabular data, which faces a set of unique
        challenges. Mainly, the presence of both continuous and discrete columns
        (ordinal &amp; discrete), and highly-imbalanced category. I don&apos;t
        really understand the solution.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.00503.pdf">https://arxiv.org/pdf/1907.00503.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generalization in Generative Adversarial Networks: A Novel Perspective
        from Privacy Protection</td>
      <td style="text-align:left">
        <p>This papers is based on the intuition that the objection of generalization
          and privacy-protecting is one and the same i.e. to gain knowledge on an
          entire population without memorizing the features of individual members.</p>
        <p>Membership attackers i.e given a individual record and a black-box model,
          determine if the individual record has been used to train the model.</p>
        <p>This paper is indirectly sharing my philosophy in pushing for generalization.
          It also makes a point on modal-collapse of GANs is related to the memorization
          of training data and therefore, is anti-privacy perserving. Hence, introducing
          privacy-preserving network like bayesian GANs would mitigate the problem
          of modal collapse.</p>
        <p>Paper also shows that techniques that enforced Lipschitz constraints,
          also prevented information leak from membership attacks.</p>
        <p>Great paper that excels in explaining what they are doing.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1908.07882.pdf">https://arxiv.org/pdf/1908.07882.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Reducing Noise in GAN Training with Variance Reduced Extragradient</td>
      <td
      style="text-align:left">
        <p>This paper is GANs is especially sensitive to noise due to its conflicting
          minimax objective. The authors shows in general, single objective DNN always
          points towards the optimal solution. However for GANs, noisy gradient can
          throw the network off-course. Stochasticity Breaks Extragradient</p>
        <p>The authors also proof that for mini-batch size less than half the dataset,
          standard stochastic optimization breaks.</p>
        <p>Implementation details in Appendix is critical. However, I do not see
          myself implementing this. Hence, my takeaway from this paper is to increase
          mini-batch size.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1904.08598.pdf">https://arxiv.org/pdf/1904.08598.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Variational inequality perspective on generative adversarial net</td>
      <td
      style="text-align:left">
        <p>This paper is in alignment with &quot;Reducing Noise in GAN Training with
          Variance Reduced Extragradient&quot; that stochastic gradient descent is
          not optimal for a two-player game such as GANs.</p>
        <p>The authors proposed three solutions, averaging, extrapolation and extrapolation
          from the past. If I am not wrong the first two solution have already been
          done. So, the enhancement that this paper brings about is its reasoning
          from the perspective of variational inequality problems (VIP).</p>
        <p>Saturating GANs, minimizing the odds of generated images as fake.</p>
        <p>Non-saturating GANs, maximizing the odds of generated images as real.
          In the latter case, I do not understand the authors claims that it is not
          a zero-sum game, because there are no other possibilities.
          <br />Even on such a small dataset, the improvement is small.
          <br />My takeaway is averaging gradient is a simplest and most effective approach.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1802.10551.pdf">https://arxiv.org/pdf/1802.10551.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Self-Defense for Cycle-Consistent GANs</td>
      <td style="text-align:left">
        <p>So many typos errors in this paper.</p>
        <p></p>
        <p>This paper presents a interesting problem. It seems that Cycle-GANs are
          ignoring much segmentation information when translating from image -to-semantic
          and back from semantic-to-image. In some cases, when image-to-semantic
          map translation fails with capture key objects such as a car, the semantic-to-image
          translation process will still generate a vehicle despite having no semantic
          signal for a car i.e. the network is hallucinating.</p>
        <p></p>
        <p>The authors named this phenomenon as self-adversarial attack. Experiments
          show that tiny perturbation in the semantics will cause the network to
          output garbage, as the low-amplitude structural that the network relies
          on to &apos;cheat&apos; has been destroyed.
          <br />Three ways to defend against this attack is proposed. Adding noise to
          disturb structural noise during training, the use of guess estimator to
          differentiate between reconstructed and original image, and reducing weightage
          of richer domain.</p>
        <p>The result is a network that produces a more &apos;honest&apos; reconstruction.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1908.01517.pdf">https://arxiv.org/pdf/1908.01517.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">The proposed solution is to introduced a adversarial auxiliary classifier
        to compete with the generator. Low diversity images from the generator
        will incur large losses in its cost function.Twin Auxiliary Classifiers
        GAN</td>
      <td style="text-align:left">
        <p>This paper bring up a fact about ACGANs, where as the number of class
          increases the diversity decreases. This fact is in alignment with &quot;Mode
          Seeking Generative Adversarial Networks for Diverse Image Synthesis&quot;</p>
        <p></p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.02690.pdf">https://arxiv.org/pdf/1907.02690.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Conditional Independence Testing using Generative Adversarial Networks</td>
      <td
      style="text-align:left">Interesting paper on using GANs for causal testing. However, this is not
        my area. Should read some other time.</td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.04068.pdf">https://arxiv.org/pdf/1907.04068.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Quality Aware Generative Adversarial Networks</td>
      <td style="text-align:left">This paper really over-glorify image quality metrics like SSIM index and
        NIQE. While I am unaware of these quality metric, the novelty of this paper
        is the conversion of these metrics into a workable loss function for image
        regularization.</td>
      <td style="text-align:left"><a href="https://www.iith.ac.in/~lfovia/qagan_neurips_2019.pdf">https://www.iith.ac.in/~lfovia/qagan_neurips_2019.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Face Reconstruction from Voice using Generative Adversarial Networks</td>
      <td
      style="text-align:left">As the paper title says.. reconstruction of face using voice. Image quality
        is bad. Perhaps there is a way to get better image quality from only voice.</td>
        <td
        style="text-align:left"><a href="https://arxiv.org/pdf/1905.10604.pdf">https://arxiv.org/pdf/1905.10604.pdf</a>
          </td>
    </tr>
    <tr>
      <td style="text-align:left">Learning from Label Proportions with Generative Adversarial Networks</td>
      <td
      style="text-align:left">
        <p>Label Proportion problem (LLP) is when the training data is grouped into
          bags, and only the proportion of each bag is known.</p>
        <p>The generator will produce fake bags w/ correct proportion to fool the
          discriminator. There is a cross entropy loss term in the discriminator
          to encourage the generator to learn the correct label .</p>
        <p>Results in this papers show that the larger the bag size, the greater
          the error. Nonetheless, the network performs only marginally worse compared
          to CNN trained on directly. ground truth labels. Knowing only proportions
          is a huge handicap, but this network performs well.</p>
        <p>The conclusion of paper is interesting.</p>
        <p>&quot;Nevertheless, limitations in our method can be summarized in the
          following three aspects. Firstly,learning complexity in the sense of PAC
          has not been involved in this study. That is to say, we cannotevaluate
          the performance under limited data. Secondly, there is no guarantee on
          algorithm robustnessto data perturbations, notably when the proportions
          are imprecisely provided. Thirdly, other GANs(such as WGAN [3]) are not
          considered in our current model and their performance is unknown.&quot;</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1909.02180.pdf">https://arxiv.org/pdf/1909.02180.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">The numerics of GANs</td>
      <td style="text-align:left">
        <p>This paper unlike other similar papers is not able proving the existence
          of a Nash Equalibrium. Instead, it is more handling the practical issues
          that arises such computation and numerics difficulties.</p>
        <p>The authors propose consensus optimization as an alternative to simultaneous
          gradient ascent.</p>
        <p>v(x) is the direction of the gradient.</p>
        <p>Consensus use 0.5 |v(x)| terms to calculate gradient vector for regularization.</p>
        <p>Not sure about details. Dont understand the proof.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1705.10461.pdf">https://arxiv.org/pdf/1705.10461.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">HexaGAN: Generative Adversarial Nets for Real World Classification</td>
      <td
      style="text-align:left">
        <p>HexGAN tries to solve 3 common problems in real-world data; incomplete
          data points, imbalance data set and missing labels.</p>
        <p>For which, the authors proposed 3 solution; imputing missing elements,
          conditional generation, and pseudo-labels.</p>
        <p>From my understanding, all 3 solutions are not novel i.e. they have been
          studied in isolation. However, this paper is the first to bring all three
          together to handle real-world data set.
          <br />The experimental is only on MNIST... how can that even be considered &quot;REAL
          WORLD&quot; data? I smell horse shit.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1902.09913.pdf">https://arxiv.org/pdf/1902.09913.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">HyperGAN: A Generative Model for Diverse, Performant Neural Networks</td>
      <td
      style="text-align:left">
        <p>Interestingly I found out from this paper that dropout layer was formulated
          as a Bayesian approach and are equivalent to a deep Gaussian process. That
          being said, the main purpose of HyperGAN is to obtain uncertainty estimate,
          rather than just given an maximum-likehood accuracy.</p>
        <p></p>
        <p>From my understanding, this architecture is using GANs to generate the
          weights of the target classifier network. Hence, the target network is
          no longer trained by back-propagation. I can see two benefits of training
          the network this way. Firstly, this training method should be privacy-perserving,
          and secondly, model training is regularized by an adversarial learning
          maintain &quot;exploratory energy&quot; by aligning correlated latent code
          with a high-entropy prior. As a results, model trained using this method
          have more diverse feature maps.</p>
        <p></p>
        <p>It is interesting to see how the model performs in different data set.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1901.11058.pdf">https://arxiv.org/pdf/1901.11058.pdf</a>.</td>
    </tr>
    <tr>
      <td style="text-align:left">A Large-Scale Study on Regularization and Normalization in GANs</td>
      <td
      style="text-align:left">
        <p>This paper is a empirical study of GANs, involving various loss functions,
          architectures and regularization techniques.</p>
        <p>The best regularization method is spectral norm as it is the most reliable.
          Though, gradient penalty-5 and layer norm works well too i.e. better than
          vanilla.</p>
        <p>As for the loss functions, there is no clear winner. But, spectral norm
          is seems to be the most reliable.</p>
        <p>For architecture, SNDCGAN outperforms resnet-19 in general. However, SN
          &amp; tuned-GP worked consistently to improve performance compared to baseline.
          The downside of GP is the need to tune a hyper-parameter.</p>
        <p>Interesting GP + SN collectively improve the FID more significantly.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1807.04720.pdf">https://arxiv.org/pdf/1807.04720.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Kernel Mean Matching for Content Addressability of GANs</td>
      <td style="text-align:left">
        <p>This GAN allow users to specify desired images based on similar samples
          generated before, without retraining. Interesting this method is a form
          of unconditional implicit model.</p>
        <p>
          <br />The question now is how can the model take an input without conditioning?
          The answer is performances gradient descent using latent noise z, as the
          parameter, such that the extracted features of G(z) matches the features
          of the input image. Note that: the features extractor is not part of the
          GAN and is another separate component.</p>
        <p></p>
        <p>I do not understand the details of the feature extractor. It is cool that
          this paper has a google collab page. 10/10 results reproducibility. I should
          learn from the authors..</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.05882.pdf">https://arxiv.org/pdf/1905.05882.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">MetricGAN: Generative Adversarial Networks based Black-box Metric Scores
        Optimization for Speech Enhancement</td>
      <td style="text-align:left">
        <p>This paper is about optimizing a metric, while using adversarial loss
          to train GAN. The question is, why do we need adversarial learning when
          learning objective is not intractable? One reason I can come up with is
          the selected metric is simply a heuristic on indication of an intractable
          learning objective. In that case, the heuristic metric can point the network
          in a &apos;general&apos; direction as a form of regularization. The problem
          is the blind optimization of a KPI/metric can hurt the overall performance.
          Because in this study, the quality metric has been used as a loss function,
          a qualitative test was conducted with human listerner using A/B testing.
          I think the way that this paper to judge audio quality is very dubious.</p>
        <p></p>
        <p>Interestingly speech can be decomposed analyzed as an image using spectrogram.
          As such, GANs can be used to generate audio of a specific speaker.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.04874.pdf">https://arxiv.org/pdf/1905.04874.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Entropic GANs meet VAEs: A Statistical Approach to Compute Sample Likelihoods
        in GANs</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1810.04147.pdf">https://arxiv.org/pdf/1810.04147.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Self-Attention Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1805.08318.pdf">https://arxiv.org/pdf/1805.08318.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Non-Parametric Priors For Generative Adversarial Networks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.07061.pdf">https://arxiv.org/pdf/1905.07061.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generative Adversarial User Model for Reinforcement Learning Based Recommendation
        System</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1812.10613.pdf">https://arxiv.org/pdf/1812.10613.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Finding Mixed Nash Equilibria of Generative Adversarial Networks</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1811.02002.pdf">https://arxiv.org/pdf/1811.02002.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Multi-objective training of Generative Adversarial Networks with multiple
        discriminators</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1901.08680.pdf">https://arxiv.org/pdf/1901.08680.pdf</a>
      </td>
    </tr>
  </tbody>
</table>### Others

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
      <td style="text-align:left">Color Indexing</td>
      <td style="text-align:left">
        <p>Histogram Intersection with real-time indexing of stored models for identification.
          Histogram back-propagation.</p>
        <p>Colors as implicit cues to object identity vs. explicit cues from geometric
          features.</p>
        <p>Colors are unreliable due to sensitivity to lighting conditions, but some
          works have addressed this.
          <br />Different parts of our brain, identifies and locate images (Where vs.
          What).
          <br />Color histogram are insensitive to translation, rotation and occlusion.
          (Try experiment to validate this claim)</p>
        <p>Incremental Intersection for efficient indexing (what is it?)</p>
        <p>Histogram back-propagation. (Where is it?)</p>
        <p>cumulative histogram</p>
        <p>fraction of the multidimensional space (representation learning. chance
          of different model being close is low)</p>
      </td>
      <td style="text-align:left"><a href="https://link.springer.com/article/10.1007/BF00130487">https://link.springer.com/article/10.1007/BF00130487</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Texture Synthesis by Non-parametric Sampling</td>
      <td style="text-align:left">Paper is based on a problem that texture synthesis are either periodic
        or completely random. This paper introduce way to synthesis periodic textures
        with randomized component.</td>
      <td style="text-align:left"><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf">https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Examples Are Not Bugs, They Are Features</td>
      <td style="text-align:left">
        <p>Very interesting paper. This paper shows that network that trains on adversarial
          cat images (actual class is dog) can still reliable classify cat images.
          Hence, the authors claim &quot;Adversarial vulnerability is a direct result
          of our models&#x2019; sensitivity to well-generalizing features in the
          data.&quot;</p>
        <p></p>
        <p>Not sure the exact definition of robust vs non-robust features (I do have
          a general idea).
          <br />
        </p>
        <p>In my own words, adversarial examples comes from our chase for higher
          accuracy. Therefore, the network learn to use highly predictive features
          regardless of its robustness. However, most of these non-robust features
          are not detectable by humans, but obvious to machines. Hence, in order
          to increase robustness, we need to train the network with human priors.</p>
        <p></p>
        <p>Conclusion in Paper: &quot;Overall, attaining models that are robust and
          interpretable will require explicitly encoding human priors into the training
          process.&quot;</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.02175.pdf">https://arxiv.org/pdf/1905.02175.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial
        Training</td>
      <td style="text-align:left">
        <p>The concept of label leaking is when the network is able to classify adversarial
          examples better than the original class examples. This phenomenon happens
          when adversarial training is employed to increase the robustness of a network
          against adversarial attacks. The irony here is that now the network performs
          well only if it is under-attack.</p>
        <p>According to this paper, the data manifold of adversarial examples shifts
          and deviates away from the original samples. This reminds me of domain
          adaptation.</p>
        <p>The authors proposed feature scattering i.e. maximizing the distance between
          features between samples, such that the network will not only move away
          from the decision boundary, but also maintain some distance between each
          other. Hence, increasing robustness.</p>
        <p>Not really sure about implementation details.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1907.10764.pdf">https://arxiv.org/pdf/1907.10764.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Fooling Neural Network Interpretations via Adversarial Model Manipulation</td>
      <td
      style="text-align:left">
        <p>This paper is about attacking the interpretation of a network instead
          of its accuracy. In fact, the accuracy of the perturbed network is the
          remains within 1% of the original network.</p>
        <p>The attack encourage the network to make use of irrelevant information
          with strong predictive capability.</p>
        <p></p>
        <p>The types of passive interpretation fooling are location fooling i.e.
          always make a certain area of a image important, center-of-mass fooling
          i.e. make the an area that is far from the original salience important,
          and k-top fooling i.e. reduce the importance of top k% of pixels. Another
          kind of fooling is called active fooling, which actively makes another
          object in the scene important.</p>
        <p>There is no solution proposed to stop interpretation fooling.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1902.02041.pdf">https://arxiv.org/pdf/1902.02041.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial training for free!</td>
      <td style="text-align:left">
        <p>Adversarial training is slow, especially on large data set. This paper
          is about speeding up adversarial training by re-using parts of the gradient
          information.</p>
        <p></p>
        <p>Targeted vs Non-Targeted. Shifting a classification from natural class
          to another specified class versus shifting classification away from natural
          class.</p>
        <p></p>
        <p>No sure about exact implementation, or algorithm.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1904.12843.pdf">https://arxiv.org/pdf/1904.12843.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Adversarial Training and Robustness for Multiple Perturbations</td>
      <td
      style="text-align:left">
        <p>Models trained against a single type of adversarial attack, is only robust
          against that particular form of perturbation.</p>
        <p>On contrary, an ensemble of two models, each trained against one form
          of attack, could be vulnerable to both.
          <br />This paper is a study of defense again multi-prong attacks.</p>
        <p>Paper is too difficult for me. I am not familiar with many of the basic
          terminology.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1904.13000.pdf">https://arxiv.org/pdf/1904.13000.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Detecting Overfitting via Adversarial Examples</td>
      <td style="text-align:left">
        <p>This paper discussed a problems that I brought up to my supervising professor.
          The problem of the over-fitting to test set that happen naturally as observe
          test data indirectly through our experiment.</p>
        <p>Also, the papers also states a point about a distribution shift problem
          when an alternative test set is being used. I believe is actually the domain
          adaptation problem i.e. domain shift from data set bias and covariance
          shift. Cifar10.1 follows the collecting procedure of Cifar10, but still
          experience distribution shift regardless, the authors are hinting that
          using alternative test data is not feasible. However, I believe that using
          adversarial examples will also shift the domain, as seen in previous experiment
          where the adversarial examples are moved towards toward the negative class.</p>
        <p></p>
        <p>The authors use some math on &quot;importance weighted risk estimate&quot;
          that I do not understand to detect over-fitting.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1903.02380.pdf">https://arxiv.org/pdf/1903.02380.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Certified Adversarial Robustness with Addition Gaussian Noise</td>
      <td
      style="text-align:left">
        <p>The idea of this paper is simple. Adding gaussian noise to drown any adversarial
          perturbation, making any adverse signal undetectable to the network.</p>
        <p>From there, the authors proof that error is bounded. Higher sigma of Gaussian
          increases robustness, while increasing the difference between 1st and 2nd
          highest due to accuracy drop. Therefore, optimal sigma is not obvious.</p>
        <p>My takeaway: Choose highest sigma, where maximum difference between 1st
          and 2nd highest probability. Optimal sigma is obtain with experiment.</p>
        <p>Interesting this work has not bound the network to the &quot;size&quot;
          of attack L. Instead, error is bounded only on sigma and a known L i.e.
          any attack with L less than a range of sigma would be bounded.</p>
        </td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1809.03113.pdf">https://arxiv.org/pdf/1809.03113.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Unlabeled Data Improves Adversarial Robustness</td>
      <td style="text-align:left">This paper states that the robustness of a network is proportional to
        the size of dataset. However, having access to high-quality labels for
        large number of images can be a challenge. Thus, the authors attempt improve
        robustness without labels, through methods of semi-supervised learning
        with the use of pseudo-labels.
        <br />Unlabeled data is labelled as the 11th class when combined with Cifar10</td>
      <td
      style="text-align:left"><a href="https://arxiv.org/pdf/1905.13736.pdf">https://arxiv.org/pdf/1905.13736.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">You Only Propagate Once: Accelerating Adversarial Training via Maximal
        Principle</td>
      <td style="text-align:left">This paper is about the discovery that adversarial perturbation mainly
        affects the first layers of the network. Hence, restricting updates to
        only one layer greatly speeds of training with adversarial examples. I
        do not understand the proof.</td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.00877.pdf">https://arxiv.org/pdf/1905.00877.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Metric Learning for Adversarial Robustness</td>
      <td style="text-align:left">
        <p>This paper makes use of the triplet loss framework to improve adversarial
          robustness. The idea is &apos;Our motivation is that the triplet loss function
          will pull all the images of one class,both natural and adversarial, closer
          while pushing the images of other classes far apart. Thus, an image and
          its adversarial counterpart should be on the same manifold, while all the
          members of the&#x2018;false&#x2019; class should be forced to be separated
          by a large margin.&apos; Note that the authors is not using the standard
          triplet loss.</p>
        <p>This paper shows that simple enhancement to existing works can achieve
          good results.</p>
      </td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1909.00900.pdf">https://arxiv.org/pdf/1909.00900.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Generalized No Free Lunch Theorem for Adversarial Robustness</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1810.04065.pdf">https://arxiv.org/pdf/1810.04065.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Are Generative Classifiers More Robust to Adversarial Attacks?</td>
      <td
      style="text-align:left"></td>
        <td style="text-align:left"><a href="https://arxiv.org/pdf/1802.06552.pdf">https://arxiv.org/pdf/1802.06552.pdf</a>
        </td>
    </tr>
    <tr>
      <td style="text-align:left">Transferable Adversarial Training: A General Approach to Adapting Deep
        Classifiers</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="http://proceedings.mlr.press/v97/liu19b/liu19b.pdf">http://proceedings.mlr.press/v97/liu19b/liu19b.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left">Simple Black-box Adversarial Attacks</td>
      <td style="text-align:left"></td>
      <td style="text-align:left"><a href="https://arxiv.org/pdf/1905.07121.pdf">https://arxiv.org/pdf/1905.07121.pdf</a>
      </td>
    </tr>
    <tr>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
    </tr>
  </tbody>
</table>