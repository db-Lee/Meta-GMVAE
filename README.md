# Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning
This is the Pytorch implementation for the paper ["**Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning**", in ICLR 2021.](https://openreview.net/pdf?id=wS0UFjsNYjn)


## Abstract
<img align="middle" width="500" src="https://github.com/db-Lee/Meta-GMVAE/blob/main/concept.png">
Unsupervised learning aims to learn meaningful representations from unlabeled data which can capture its intrinsic structure, that can be transferred to downstream tasks. Meta-learning, whose objective is to learn to generalize across tasks such that the learned model can rapidly adapt to a novel task, shares the spirit of unsupervised learning in that the both seek to learn more effective and efficient learning procedure than learning from scratch. The fundamental difference of the two is that the most meta-learning approaches are supervised, assuming full access to the labels. However, acquiring labeled dataset for meta-training not only is costly as it requires human efforts in labeling but also limits its applications to pre-defined task distributions. In this paper, we propose a principled unsupervised meta-learning model, namely Meta-GMVAE, based on Variational Autoencoder (VAE) and set-level variational inference. Moreover, we introduce a mixture of Gaussian (GMM) prior, assuming that each modality represents each class-concept in a randomly sampled episode, which we optimize with Expectation-Maximization (EM). Then, the learned model can be used for downstream few-shot classification tasks, where we obtain task-specific parameters by performing semi-supervised EM on the latent representations of the support and query set, and predict labels of the query set by computing aggregated posteriors. We validate our model on Omniglot and Mini-ImageNet datasets by evaluating its performance on downstream few-shot classification tasks. The results show that our model obtains impressive performance gains over existing unsupervised meta-learning baselines, even outperforming supervised MAML on a certain setting.


__Contribution of this work__
- We propose a novel unsupervised meta-learning model, namely Meta-GMVAE, which meta-learns the set-conditioned prior and posterior network for a VAE. Our Meta-GMVAE is a principled unsupervised meta-learning method, unlike existing methods on unsupervised meta-learning that combines heuristic pseudo-labeling with supervised meta-learning.
- We propose to learn the multi-modal structure of a given dataset with the Gaussian mixture prior, such that it can adapt to a novel dataset via the EM algorithm. This flexible adaptation to a new task, is not possible with existing methods that propose VAEs with Gaussian mixture priors for single task learning.
- We show that Meta-GMVAE largely outperforms relevant unsupervised meta-learning baselines on two benchmark datasets, while obtaining even better performance than a supervised meta-learning model under a specific setting. We further show that Meta-GMVAE can generalize to classification tasks with different number of ways (classes).


## Dependencies
This code is written in Python. Dependencies include
* python >= 3.6
* pytorch = 1.4 or 1.7
* tqdm

## Data
* Download Omniglot data from [here](https://drive.google.com/file/d/1aipkJc4JDj91KuiI_VuHj752rdmNXyf_/view?usp=sharing). 
* Download pretrained features for Mini-ImageNet from [here](https://drive.google.com/file/d/1NKYDSHEIQgeTlcrB37ZOZ40N309vcNT8/view?usp=sharing).
* (Optional) If you want to train SimCLR from scratch, download images for ImageNet from [here](https://drive.google.com/file/d/1p7Rd59AtM2Faldzv-ju934zPeJuVXqGh/view?usp=sharing).

data directory should be look like this:
```shell
data/

├── omiglot/
  ├── train.npy
  ├── val.npy
  └── test.npy
  
├── mimgnet/
  ├── train_features.npy
  ├── val_features.npy
  └── test_features.npy
  
└── imgnet/ -> (optional) if you want to train SimCLR from scratch
  ├── images/
    ├── n0210891500001298.jpg  
    ├── n0287152500001298.jpg 
	       ...
    └── n0236282200001298.jpg 
  ├── train.csv
  ├── val.csv
  └── test.csv
```

## Experiment
To reproduce **Omniglot 5-way experiment** for Meta-GMVAE, run the following code:
```bash
cd omniglot
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/omniglot/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/omniglot-5way-experiment) --way 5 --sample-size 200
```

To reproduce **Omniglot 20-way experiment** for Meta-GMVAE, run the following code:
```bash
cd omniglot
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/omniglot/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/omniglot-20way-experiment) --way 20 --sample-size 300
```

To reproduce **Mini-ImageNet 5-way experiment** for Meta-GMVAE, run the following code:
```bash
cd mimgnet
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/mimgnet/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/mimgnet-5way-experiment)
```

(Optional) To reproduce SimCLR features for Mini-ImageNet, run the following code:
```bash
cd simclr
python main.py --data-dir DATA DIRECTORY (e.g. /home/dongbok/data/imgnet/) --save-dir SAVE DIRECTORY (e.g. /home/dongbok/simclr-experiment) --feature-save-dir FEATURE SAVE DIRECTORY (e.g. /home/dongbok/data/mimgnet)
```

## Reference
To cite the paper, please use this BibTex
```bibtex
@inproceedings{lee2021metagmvae,
  title={Meta-GMVAE: Mixture of Gaussians VAEs for Unsupervised Meta-Learning},
  author={Dong Bok Lee and Dongchan Min and Seanie Lee and Sung Ju Hwang},
  booktitle={Proceedings of the 9th International Conference on Learning Representations},
  year={2021}
}
```
