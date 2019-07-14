## Phylogeny-regularized Convolutional NeuralNetwork
With the development and decreasing cost of next-generation sequencing technologies, the study of the human microbiome has become an important research field accounting for huge potentials in clinical applications such as drug response predictions, patient stratification, and disease diagnosis. Thus, it is essential and desirable to build a microbiome-based predictor for clinical outcomes driven by microbiome profiles consisting of taxa abundance and a phylogenetic tree. One important characteristic is all microbial species are not uniformly distributed on the phylogenetic tree but tend to be clustered at different phylogenetic depths. Thus, the phylogenetic tree provides for the structure of microbiome,which might be an important prior information for prediction. However, prediction methods that consider the phylogenetic tree in a rigorous way are under-developed. We develop a deep learning prediction method "Phylogeny-regularized convolutional Neural Network,"(pCNN) for microbiome-based prediction. The advantage of pCNN is that it uses the convolutional kernel to capture the signals of microbiome species with close evolutionary relationship in a local receptive field. Moreover, pCNN uses different convolutional layer to capture different taxonomic rank (e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced by the phylogenetic tree.

![Figure](https://github.com/alfredyewang/pCNN/blob/master/docs/Architecture.jpg?raw=true)

For details of pCNN, users can refer to our paper "**A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based Predictionsn**".


## Requirements

- Python 3.6
- TensorFlow == 1.12.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3

```
 pip3 install -r requirements.txt
```

## Installation



## Example
### USA Human Gut Microbiome data
```
 python3 pCNN_conitnous.py data/USA
```

### Malawiantwin pairs Human Gut Microbiome data

```
 python3 pCNN_binary.py data/Malawiantwin_pairs
```
