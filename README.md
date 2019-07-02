## Phylogeny-regularized Convolutional NeuralNetwork
We develop a deep learning prediction method "Phylogeny-regularized convolutional NeuralNetwork,"(pCNN) for microbiome-based prediction. The advantage of pCNN is that it uses the convolutionalkernel to capture the signals of microbiome  species  with close  evolutionary relationship in a localreceptive field. Moreover, pCNN uses different convolutional layer to capture different taxonomic rank(e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced bythe phylogenetic tree.

For details of pCNN, users can refer to our paper "**A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based Predictionsn**".

## Requirements

- Python 3.6 (64-bit)
- TensorFlow == 1.12.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn == 0.20.3

## Example
### USA Human Gut Microbiome
- Train

### Malawiantwin pairs Human Gut Microbiome
