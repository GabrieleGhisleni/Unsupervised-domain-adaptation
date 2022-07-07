# UniTN - DeepL project course - Unsupervised Domain Adaptation
in the UDA setup we have labelled data source but we are not interested in the performance in the source domain but only in the performance of the model in a different target domain where instead the data are not annotated.

$$D_{S} = \{(x_{i}^{S}, y_{i}^{S})\}_{i=1}^{N}$$ 
$$D_{T} = \{(x_{j}^{T})\}_{j=1}^{M}$$

we implemented three main approaches: Discrepancy-based methods (MMD), Adversarial-based methods (GradReversal), Reconstruction-based methods (DRCN).

## Discrepancy-based methods (MMD)
These methods are also called deep domain confusion, the main idea is that we force the network to produce two similar feature representation of the two different domain. We developed the Maximum Mean Discrepancy (MMD) to achieve this, paper: https://arxiv.org/pdf/1412.3474.pdf.


## Adversarial-based methods (GradReversal)
Generally involve a domain discriminator to enforce domain confusion (very much like how GAN works). As the training progresses, the approach promotes the emergence of “deep” features that are (i) discriminative for the main learning task on the source domain and (ii) invariant with respect to the shift between the domains. We do this implementing the following paper <i>Unsupervised Domain Adaptation by Backpropagation</i> at https://arxiv.org/pdf/1409.7495.pdf.

## Reconstruction-based methods (DRCN)
The model called Deep Reconstruction Classification Network (DRCN), following: https://arxiv.org/pdf/1607.03516v2.pdf which jointly learns a shared encoding representation for two tasks: i) supervised classification of labeled source data, and ii) unsupervised reconstruction of unlabeled target data. In this way, the learnt representation not only preserves discriminability, but also encodes useful information from the target domain. 
