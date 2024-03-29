- **Scaling Up Graph Neural Networks Via Graph Coarsening.** *Zengfeng Huang, Shengzhong Zhang, Chong Xi, Tang Liu, Min Zhou.* ICML 2021.
  - resources: [paper](https://arxiv.org/pdf/2106.05150v1), [code](https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening)
  - contributions:
    - Train GNNs on a coarsened graph, which can be obtained from off-the-shelf coarsening methods, and transfer the learned model to the node classification task. Avoid sampling and allow full-batch training.
    - The proposed method can be viewed as learning low dimensional subspace representation with sparsity constraints imposed by some pre-defined norm.
- **Approximate Graph Propagation.** *Hanzhi Wang, Mingguo He, Zhewei Wei, Sibo Wang, Ye Yuan, Xiaoyong Du, Ji-Rong Wen.* KDD 2021.
  - resources: [paper](https://arxiv.org/pdf/2106.03058)
  - contributions:
    - Define a general propagation matrix for GNNs in the form of infinite series that allows flexible combinations of multi-hop information.
    - Approximate the infinite series with its partial sum under a given error tolerrance.
    - Propose an iterative algorithm to calculate the partial sum, and further propose a randomized version to reduce the computational cost with controlled error.
    - The propagation procedure is linear of the node features, and therefore, the back-propagation is similar to typical propagation-based GNNs. The additional computational cost lies in pre-computing the propagation matrix.
- **Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs.** *Jialin Dong, Da Zheng, Lin F. Yang, Geroge Karypis.* KDD 2021.
  - resources: [paper](https://arxiv.org/pdf/2106.06150)
  - contributions:
    - Point out that the bottleneck of training GNNs over large graphs lies in copying data from CPU memory to GPU memory.
    - Maintain a cache in GPU memory to store a small number of most probably visited node embeddings, and aggregate these embeddings weighted by their importance.
    - Provide error analysis of the proposed method.
- **GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings.** *Matthias Fey, Jan E. Lenssen, Frank Weichert, Jure Leskovec.* ICML 2021.
  - resouces: [paper](https://arxiv.org/pdf/2106.05609), [code](https://github.com/rusty1s/pyg_autoscale)
  - contributions:
    - Point out that sampling based methods for scability inevitably lose expressiveness.
    - Introduce extra memroy to maintain historical embeddings to approximate those out of batch, and prove the error bounds with an assumption of Lipschitz continuity.
    - Empirically show that the performance is comparable with full-batch training methods, while the memory consumption is on par with sampling based methods.
    - Make public an efficient PyG implementation of the proposed method.
- **Scalable Graph Neural Networks via Bidirectional Propagation.** *Ming Chen, Zhewei Wei, Bolin Ding, Yaliang Li, Ye Yuan, Xiaoyong Du, Ji-Rong Wen.* NeurIPS 2020.
  - resources: [paper](https://papers.nips.cc/paper/2020/file/a7789ef88d599b8df86bbee632b2994d-Paper.pdf), [code](https://github.com/chennnM/GBP)
  - contributions:
<!-- - **SIGN: Scalable Inception Graph Neural Networks.** *Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, Federico Monti.* NeurIPS 2020.
  - resources: [paper](), [code](https://github.com/twitter-research/sign)
  - contributions: -->
- **Bandit Samplers for Training Graph Neural Networks.** *.* NeurIPS 2020.
  - resources: [paper](https://papers.nips.cc/paper/2020/file/4cea2358d3cc5f8cd32397ca9bc51b94-Paper.pdf)
  - contributions:
- **GraphSAINT: Graph Sampling Based Inductive Learning Method.** *Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, Viktor Prasanna.* ICLR 2020.
  - resources: [paper](https://openreview.net/pdf?id=BJe8pkHFwS), [review](https://openreview.net/forum?id=BJe8pkHFwS), [code](https://github.com/GraphSAINT/GraphSAINT)
  - contributions:
- **Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks.** *Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, Quanquan Gu.* NeurIPS 2019.
  - resources: [paper](https://papers.nips.cc/paper/9303-layer-dependent-importance-sampling-for-training-deep-and-large-graph-convolutional-networks.pdf), [code](https://github.com/acbull/LADIES)
  - contributions:
- **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.** *Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh.* KDD 2019.
  - resources: [paper](https://dl.acm.org/doi/pdf/10.1145/3292500.3330925), [code](https://github.com/benedekrozemberczki/ClusterGCN)
  - contributions:
- **Adaptive Sampling Towards Fast Graph Representation Learning.** *Wenbing Huang, Tong Zhang, Yu Rong, Junzhou Huang.* NeurIPS 2018.
  - resources: [paper](https://papers.nips.cc/paper/2018/file/01eee509ee2f68dc6014898c309e86bf-Paper.pdf), [code](https://github.com/huangwb/AS-GCN)
  - contributions:
- **Stochastic Training of Graph Convolutional Networks with Variance Reduction.** *Jianfei Chen, Jun Zhu, Le Song.* ICML 2018.
  - resources: [paper](http://proceedings.mlr.press/v80/chen18p/chen18p.pdf), [code](https://github.com/thu-ml/stochastic_gcn)
  - contributions:
- **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.** *Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018.
  - resources: [paper](https://openreview.net/pdf?id=rytstxWAW), [review](https://openreview.net/forum?id=rytstxWAW), [code](https://github.com/matenure/FastGCN)
  - contributions: