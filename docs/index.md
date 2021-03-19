# ECE-GY 9123: Deep Learning

## Course Project: Contrastive Visual Representation Learning Techniques

### ©️ Cheng Chen(cc6858@nyu.edu), Yimiao Zhang(yz6756@nyu.edu)

## Plan

### Problem Statement

### Literature Survey

### Datasets

### Models

### Goals and Deliverables

## Proposal

### Problem Statement

Self-supervised learning (SSL) is a form of unsupervised learning where we try to form a supervised learning task automatically from unlabeled data. Contrastive learning is a general SSL approach where visual representations (i.e., features) are learned by contrasting positive and negative examples, and can further be transferred to downstream tasks including detection and segmentation by fine-tuning. Unsupervised representation learning is highly successful in NLP as shown by BERT, while supervised pre-training is still dominant in computer vision, where unsupervised methods generally lag behind. Although contrastive learning is not a new paradigm, it has led to great empirical success in computer vision tasks with unsupervised contrastive pre-training.

### Literature Survey

***[MoCo][2]*** views contrastive learning as training an encoder for a dictonary look-up task where the dictionary is discrete on high-dimensional continuous inputs such as images or patches. Suppose an encoded image query $q$ matches a single key $k_+$ in dictionary. The contrastive loss is considered low when $q$ is similar to its positive key $k_+$ and dissimilar to all other keys. ***MoCo*** encodes the new keys on-the-fly by a momentum updated encoder, and maintains a queue of keys as the dynamic training dictionary to handle the actual much larger dictionary (e.g., billion-scale).

***[SimCLR][3]*** is a simple framework for contrastive learning of visual representations. Two separate data augmentation operators are sampled from the same family of augmentations and applied to each data example to obtain two correlated views. A base encoder network $f(\cdot)$ and a projection head $g(\cdot)$ are trained to maximize agreement using a contrastive loss. $f(\cdot)$ encodes the view and $g(\cdot)$ projects that representation to loss space. After training is completed, the projection head $g(\cdot)$ is thrown away while encoder $f(\cdot)$ and representation $h$ are used for downstream tasks.

### Goals and Deliverables

Our baseline goal is to have a deep and thorough understanding of contrastive visual representation learning techniques, their inner nature, similarities, pros and cons, recent progress and powerful potential. We expect to implement several instantiations of recent frameworks/mechanisms, with comparative experimental results, as well as transferring and finetuning them to several specific downstream tasks such as detection and segmentation. Ideally, we would be able to identify some issues in current frameworks and come up with some ideas to improve or future directions to explore currently undervalued self-supervised learning.

[1]: https://arxiv.org/abs/1810.04805	"BERT"
[2]: https://arxiv.org/abs/1911.05722	"MoCo"
[3]: https://arxiv.org/abs/2006.10029	"SimCLR"