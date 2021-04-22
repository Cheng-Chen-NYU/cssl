***ECE-GY 9123: Deep Learning***

## Contrastive Visual Representation Learning

©️ Cheng Chen(cc6858@nyu.edu), Yimiao Zhang(yz6756@nyu.edu)

## Project Update

#### Problem Statement

Pre-training and fine-tuning paradigm in computer vision can date back to $\text{CVPR}$ 2009 when ImageNet database was presented. Afterwards, $\text{AlexNet, VGG, Inception, ResNet}$ and so on can be pre-trained on $\text{ImageNet}$ with high classification accuracy. Further, representations from these pre-trained model can be transferred to downstream tasks such as object detection and semantic segmentation. This kind of supervised pre-training is still dominant in computer vision. Recent years, unsupervised representation learning is highly successful in natural language processing, e.g., as shown by GPT and BERT. Language tasks have discrete signal space (words, sub-word units, etc.) while the raw signal of vision tasks is in a continuous, high dimensional space[^1]. Learning effective visual representation without human supervision is a long-standing problem while recent studies[^1][^2][^3][^4] are converging on a central concept known as **contrastive learning** where visual representations (i.e., features) are pre-trained by contrasting positive and negative examples.

#### Literature Survey

**MoCo**[^1]  builds a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. 

**SimCLR**[^2]  obtains two correlated views for each data example. A base encoder network $f(\cdot)$ and a projection head $g(\cdot)$ are trained to maximize agreement using a contrastive loss. $f(\cdot)$ encodes the view and $g(\cdot)$ projects that representation to loss space. 

**MoCo v2**[^3] implements two of *SimCLR*’s design improvements, MLP projection head and more data augmentation, in the MoCo framework. This establishes stronger baselines that outperform *SimCLR* and do not require large training bataches.

**SimCLR v2**[^4]  uses a deeper and wider encoder network $f(\cdot)$ and a deeper projection head $g(\cdot)$  based on the original SimCLR.

#### Datasets

- $\text{CIFAR-10/CIFAR-100}$:
  - The $\text{CIFAR-10}$ dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
  - Batch $512\times3\times32\times32$
  - The $\text{CIFAR-100}$ dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the $\text{CIFAR-100}$ are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
- $\text{ImageNet ILSVRC 2012}$ (subset)
  - 1% $\rightarrow$ 12811 images
  - 10% $\rightarrow$ 128116 images
  - Original $\rightarrow$ 1.28M images: This is the ImageNet training set that has ~1.28 million images in 1000 classes. This dataset is well-balanced in its class distribution, and its images generally contain iconic view of objects.
- Canine subset of Imagenet (https://www.kaggle.com/c/dog-breed-identification/)
  - 120 breeds of dogs
- $\text{PASCAL VOC}$ `trainval` and `test` set
  - A benchmark in visual object category recognition and detection consisting of a publicly available dataset of images and annotation

#### Models

Various methods of contrastive learning can be thought of as building dynamic dictionaries where encoders perform dictionary look-up: an encoded "query" should be similar to its matching key and dissimilar to others.

$\text{MoCo}$[^1] hypothesizes that it is desirable to build dictionaries that are large and consistent. Intuitively, a larger dictionary may better sample the underlying continuous, high-dimensional visual space, while the keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent. With similarity measured by dot product, $\text{MoCo}$ considers the $\text{InfoNCE}$ contrastive loss function
$$
\mathcal L_q=-\log\frac{\exp(q\cdot k_+/\tau)}{\sum_{i=1}^K\exp(q\cdot k_i/\tau)}
$$
where $q$ is the encoded image query $x$ and $\{k_1,k_2,k_3\ldots\}$ are encoded samples/keys of the dictionary. $q$ and $k_+$ which are two random augmented version of $x$ are psudo-labeled as positive pairs. The dictionary is maintained as a queue where the current minibatch is enqueued and the oldest minibatch is removed. Dictionary size $K$ can vary from 256 to 65536 and the larger the better. Suppose the query encoder is $f_q(\cdot)$ with parameter $\theta_q$ and the key encoder $f_k(\cdot)$ with parameter $\theta_k$. $\theta_q$ is updated simply by $\text{SGD}$ while $\text{MoCo}$ does a momentum update to $\theta_k$ where
$$
\theta_k\leftarrow m\theta_k+(1-m)\theta_q
$$
$\text{SimCLR}$[^2] doesn't adopt a dictionary mechanism explicitly. Instead, by training with a large batch size such as 8192, the batch itself can function as a dictionary and each augmented data example can function as "query" or "key". More specifically, two separate data augmentation operators $t(\cdot),t'(\cdot)$ are sampled from the same family of augmentations and applied to each data example to obtain two correlated views, i.e., a positive pair. A batch of size $N$ gives us $2(N-1)$ negative examples per positive pair from both augmented views. The loss function for a positive pair of examples $(i,j)$ is defined as
$$
\ell_{i,j}=-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf 1 _{[k\ne i]}\exp(\text{sim}(z_i,z_k)/\tau)}
$$
Here, $\mathbf 1 _{[k\ne i]}\in\{0,1\}$ is an indicator function evaluating to 1 iff $k\ne i$, $\tau$ denotes a temperature parameter, $\text{sim}(u,v)=(u^Tv)/\|u\|\|v\|$ denotes the cosine similarity and 
$$
z_i=g(f(t(x_i)))
$$
where $t(\cdot)$ denotes the *data augmentation* function, $f(\cdot)$ denotes the *base encoder* which is a $\text{ResNet}$, and $g(\cdot)$ denotes the *projection head* which is a 2-layer $\text{MLP}$ to project representation to loss space. The final loss is computed across all positive pairs, both $(i, j)$ and $(j, i)$, in a batch. This loss is termed as $\text{NT-Xent}$(the normalized temparature-scaled cross entropy loss).

$\text{MoCo v2}$[^3] verifies the effectiveness of two of $\text{SimCLR}$'s design improvements by implementing them in the $\text{MoCo}$ framework --- using an $\text{MLP}$ projection head and more data augmentation. It does not require large training batches. For fair comparisons, they also study a cosine (half-period) learning rate schedule which $\text{SimCLR}$ adopts.

$\text{SimCLR v2}$[^4] adopts $\text{SimCLR}$ and improves it in three major ways.

1. Explore larger models: From $\text{ResNet-}50(4\times)$ to $\text{ResNet-}152(3\times\text{+SK})$.
2. Explore deeper projection head $g(\cdot)$: From 2-layer to 3-layer. Instead of throwing it entirely, fine-tune from the $1st$ layer of projection head.
3. Explore memory/dictionary mechanism in $\text{MoCo}$.

Authors further propose a three-step semi-supervised learning algorithm --- unsupervised pretraining of a big $\text{ResNet}$ model using $\text{SimCLR v2}$, supervised fine-tuning on a few labeled examples, and distillation with unlabeled examples for refining and transferring the task-specific knowledge.

#### Experiments

MoCo

MLP / aug+ / cos

2048 --> 512 --> 128

linear classifier

- [x] `lr=30`

#### Preliminary Results

| Config | MLP          | cos          | Symmetric    | epochs | batch | CIFAR acc. |
| ------ | ------------ | ------------ | ------------ | ------ | ----- | ---------- |
| MoCo   |              | $\checkmark$ |              | 200    | 512   | 0.767      |
| MoCo   |              | $\checkmark$ | $\checkmark$ | 200    | 512   | 0.798      |
| MoCo   | $\checkmark$ | $\checkmark$ |              |        |       |            |
|        |              |              |              |        |       |            |

<img src="/Users/julius/cvrl/200asy.png" style="zoom:59.5%;" /><img src="/Users/julius/cvrl/200sy.png" style="zoom:59.5%;" />

#### Challenges

#### Goals and Deliverables

We will mainly focus on $\text{MoCo v1&v2}$ and $\text{SimCLR v1&v2}$ papers. $\text{MoCo}$ trains on $\text{ImageNet}$ with 8 GPUs. $\text{SimCLR}$ trains its model with Cloud TPU using 32 to 128 cores depending on batch size. We have limited time and resources compared to paper authors and anticipated outcomes are

1. Have a deep and thorough understanding of contrastive visual representation learning techniques, their inner nature, similarities, pros and cons, recent progress and powerful potential

2. Simulate the Multi-GPU/TPU behavior in Colab's 1-GPU environment

3. Implement models with the core concepts from $\text{MoCo v1&v2}$ and $\text{SimCLR v1&v2}$

4. Evaluate our implementation and expect comparable results

   - $\text{kNN}$ monitor

   - ImageNet linear classification
     - Features are frozen and a supervised linear classifier is trained
     - Metric: 1-crop ($224\times224$), top-1 validation accuracy
   - Transferring to $\text{VOC}$ objection detrction
     - A Faster $\text{R-CNN}$ detector (C4-backbone) is fine-tuned end-to-end on the $\text{VOC 07+12}$ `trainval` set and evaluated on the $\text{VOC 07}$ `test` set
     - Metric: $\text{COCO}$ suite of metrics

## Project Plan

#### Problem Statement

Pre-training and fine-tuning paradigm in computer vision can date back to $\text{CVPR}$ 2009 when $\text{ImageNet}$ database was presented. Afterwards, $\text{AlexNet, VGG, Inception, ResNet}$ and so on can be pre-trained on $\text{ImageNet}$ with high classification accuracy. Further, representations from these pre-trained model can be transferred to downstream tasks such as object detection and semantic segmentation. This kind of supervised pre-training is still dominant in computer vision. Recent years, unsupervised representation learning is highly successful in natural language processing, e.g., as shown by GPT and BERT. Language tasks have discrete signal space (words, sub-word units, etc.) while the raw signal of vision tasks is in a continuous, high dimensional space[^1]. Learning effective visual representation without human supervision is a long-standing problem while recent studies[^1][^2][^3][^4] are converging on a central concept known as **contrastive learning** where visual representations (i.e., features) are pre-trained by contrasting positive and negative examples.

#### Literature Survey

**MoCo**[^1]  builds a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. 

**SimCLR**[^2]  obtains two correlated views for each data example. A base encoder network $f(\cdot)$ and a projection head $g(\cdot)$ are trained to maximize agreement using a contrastive loss. $f(\cdot)$ encodes the view and $g(\cdot)$ projects that representation to loss space. 

**MoCo v2**[^3] implements two of *SimCLR*’s design improvements, MLP projection head and more data augmentation, in the MoCo framework. This establishes stronger baselines that outperform *SimCLR* and do not require large training bataches.

**SimCLR v2**[^4]  uses a deeper and wider encoder network $f(\cdot)$ and a deeper projection head $g(\cdot)$  based on the original SimCLR.

#### Datasets

- $\text{CIFAR-10}$
  - The $\text{CIFAR-10}$ dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
- $\text{ImageNet ILSVRC 2012}$ (subset)
  - This is the ImageNet training set that has ~1.28 million images in 1000 classes. This dataset is well-balanced in its class distribution, and its images generally contain iconic view of objects.

- $\text{PASCAL VOC}$ `trainval` and `test` set
  - A benchmark in visual object category recognition and detection consisting of a publicly available dataset of images and annotation

#### Models

The base model is a $\text{ResNet}$ encoder whose last fully-connected layer (after global average pooling) has a fixed-dimensional output (128-D). This encoder takes in images/patches and produces vector representations. The representations can be frozen upon which a linear classifier can be trained. Also, the encoder can be incorporated into an object detection model and fine-tuned end-to-end.

With similarity measured by dot product, $\text{MoCo}$[^1] considers the $\text{InfoNCE}$ contrastive loss function
$$
\mathcal L_q=-\log\frac{\exp(q\cdot k_+/\tau)}{\sum_{i=1}^K\exp(q\cdot k_i/\tau)}
$$
where $q$ is the encoded image query $x$ and $\{k_1,k_2,k_3\ldots\}$ are encoded keys of the dictionary. $q$ and $k_+$ which are two random augmented version of $x$ are psudo-labeled as positive pairs. The dictionary is maintained as a queue where the current minibatch is enqueued and the oldest minibatch is removed. Dictionary size $K$ can vary from 256 to 65536 and the larger the better. Suppose the query encoder is $f_q(\cdot)$ with parameter $\theta_q$ and the key encoder $f_k(\cdot)$ with parameter $\theta_k$. $\theta_q$ is updated simply by $\text{SGD}$ while $\text{MoCo}$ does a momentum update to $\theta_k$ where
$$
\theta_k\leftarrow m\theta_k+(1-m)\theta_q
$$
$\text{SimCLR}$[^2] doesn't adopt a dictionary mechanism explicitly. Instead, by training with a large batch size such as 8192, the batch itself can function as a dictionary and each augmented data example can function as "query" or "key". More specifically, two separate data augmentation operators $t(\cdot),t'(\cdot)$ are sampled from the same family of augmentations and applied to each data example to obtain two correlated views, i.e., a positive pair. A batch of size $N$ gives us $2(N-1)$ negative examples per positive pair from both augmented views. The loss function for a positive pair of examples $(i,j)$ is defined as
$$
\ell_{i,j}=-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf 1 _{[k\ne i]}\exp(\text{sim}(z_i,z_k)/\tau)}
$$
Here, $\mathbf 1 _{[k\ne i]}\in\{0,1\}$ is an indicator function evaluating to 1 iff $k\ne i$, $\tau$ denotes a temperature parameter, $\text{sim}(u,v)=(u^Tv)/\|u\|\|v\|$ denotes the cosine similarity and 
$$
z_i=g(f(t(x_i)))
$$
where $t(\cdot)$ denotes the *data augmentation* function, $f(\cdot)$ denotes the *base encoder* which is a $\text{ResNet}$, and $g(\cdot)$ denotes the *projection head* which is a 2-layer $\text{MLP}$ to project representation to loss space. The final loss is computed across all positive pairs, both $(i, j)$ and $(j, i)$, in a batch. 

$\text{MoCo v2}$[^3] verifies the effectiveness of two of $\text{SimCLR}$'s design improvements by implementing them in the $\text{MoCo}$ framework --- using an $\text{MLP}$ projection head and more data augmentation. It does not require large training batches. For fair comparisons, they also study a cosine (half-period) learning rate schedule which $\text{SimCLR}$ adopts.

$\text{SimCLR v2}$[^4] adopts $\text{SimCLR}$ and improves it in three major ways.

1. Explore larger models: From $\text{ResNet-}50(4\times)$ to $\text{ResNet-}152(3\times\text{+SK})$.
2. Explore deeper projection head $g(\cdot)$: From 2-layer to 3-layer. Instead of throwing it entirely, fine-tune from the $1st$ layer of projection head.
3. Explore memory/dictionary mechanism in $\text{MoCo}$.

#### Goals and Deliverables

1. Have a deep and thorough understanding of contrastive visual representation learning techniques, their inner nature, similarities, pros and cons, recent progress and powerful potential

2. Simulate the Multi-GPU/TPU behavior in Colab's 1-GPU environment

3. Implement models with the core concepts from $\text{MoCo v1&v2}$ and $\text{SimCLR v1&v2}$. We will make adjustments/modifications if needed.

4. Evaluate our implementation and expect comparable results on metrics such as

   - 1-crop ($224\times224$), top-1 validation accuracy on ImageNet

- $\text{COCO}$ suite of metrics on $\text{VOC}$ objection detection

## Project Proposal

### Problem Statement

Self-supervised learning (SSL) is a form of unsupervised learning where we try to form a supervised learning task automatically from unlabeled data. Contrastive learning is a general SSL approach where visual representations (i.e., features) are learned by contrasting positive and negative examples, and can further be transferred to downstream tasks including detection and segmentation by fine-tuning. Unsupervised representation learning is highly successful in NLP as shown by BERT, while supervised pre-training is still dominant in computer vision, where unsupervised methods generally lag behind. Although contrastive learning is not a new paradigm, it has led to great empirical success in computer vision tasks with unsupervised contrastive pre-training.

### Literature Survey

***[MoCo][2]*** views contrastive learning as training an encoder for a dictonary look-up task where the dictionary is discrete on high-dimensional continuous inputs such as images or patches. Suppose an encoded image query $q$ matches a single key $k_+$ in dictionary. The contrastive loss is considered low when $q$ is similar to its positive key $k_+$ and dissimilar to all other keys. ***MoCo*** encodes the new keys on-the-fly by a momentum updated encoder, and maintains a queue of keys as the dynamic training dictionary to handle the actual much larger dictionary (e.g., billion-scale).

***[SimCLR][3]*** is a simple framework for contrastive learning of visual representations. Two separate data augmentation operators are sampled from the same family of augmentations and applied to each data example to obtain two correlated views. A base encoder network $f(\cdot)$ and a projection head $g(\cdot)$ are trained to maximize agreement using a contrastive loss. $f(\cdot)$ encodes the view and $g(\cdot)$ projects that representation to loss space. After training is completed, the projection head $g(\cdot)$ is thrown away while encoder $f(\cdot)$ and representation $h$ are used for downstream tasks.

### Goals and Deliverables

Our baseline goal is to have a deep and thorough understanding of contrastive visual representation learning techniques, their inner nature, similarities, pros and cons, recent progress and powerful potential. We expect to implement several instantiations of recent frameworks/mechanisms, with comparative experimental results, as well as transferring and finetuning them to several specific downstream tasks such as detection and segmentation. Ideally, we would be able to identify some issues in current frameworks and come up with some ideas to improve or future directions to explore currently undervalued self-supervised learning.

[1]: https://arxiv.org/abs/1810.04805	"BERT"
[2]: https://arxiv.org/abs/1911.05722	"MoCo"
[3]: https://arxiv.org/abs/2002.05709	"SimCLR"

#### References

[^1]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. CVPR, arXiv:1911.05722, 2019.
[^2]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. ICML, arXiv:2002.05709, 2020.
[^3]: Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297, 2020.
[^4]: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. NeurIPS, arXiv:2006.10029, 2020.