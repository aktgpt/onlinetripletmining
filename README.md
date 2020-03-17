# Deep metric learning and classification with online triplet mining

Pytorch implementation of triplet networks for metric learning

# Installation
 This package requires Pytorch version 1.4.0 and TorchVision 0.5.0

# Features
* GPU implementation of online triplet loss in a way similar to pytorch loss
* Implements 1-1 sampling strategy as defined in [1]
* Random semi-hard and fixed semi-hard sampling
* UMAP visualization of the results
* Implementation of training strategy to train a classifier after learning the embeddings.
* Implementation of stratified sampling strategy for the batches.
* Implemented on MNIST dataset as an example.

# Code Structure
* __`networks.py`__
    * _ConvNet_ class - base network for embedding images in vectors and getting labels
* __`loss.py`__
    * _OnlineTripletLoss_ - triplet loss class for embeddings
    * _NegativeTripletSelector_ - class for selecting the negative sample from the batch based on the sampling strategy.
* __`train.py`__
    * _TripletTrainer_ - class for training the dataset with triplet loss and a classifier after it if required.
* __`utils.py`__
    * _make_weights_for_balanced_classes_ - assign weight to every sample in dataset for batch sampling.
    * _save_embedding_umap_ - save UMAPs of the training set and test set.
* __`config.json`__ 
    - hyperparameters for the training



# References
[1] [Theoretical Guarantees of Deep Embedding Losses Under Label Noise](https://arxiv.org/pdf/1812.02676.pdf])