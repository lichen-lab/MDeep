## Phylogeny-regularized Convolutional NeuralNetwork
We develop a deep learning prediction method "Phylogeny-regularized convolutional Neural Network,"(pCNN) for microbiome-based prediction. The advantage of pCNN is that it uses the convolutional kernel to capture the signals of microbiome species with close evolutionary relationship in a local receptive field. Moreover, pCNN uses different convolutional layer to capture different taxonomic rank (e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced by the phylogenetic tree.

<center>

![Figure](https://github.com/alfredyewang/pCNN/blob/master/docs/Architecture.jpg?raw=true =800x600)

</center>  

For details of pCNN, users can refer to our paper "**A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based Predictionsn**".


## Requirements and Installation

- Python 3.6
- TensorFlow == 1.12.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3

Download pCNN and install requirements:
```
git clone https://github.com/alfredyewang/pCNN
pip3 install -r requirements.txt
```


## Example
### USA Human Gut Microbiome data
```
 python3 pCNN_conitnous.py data/USA
```

### Malawiantwin pairs Human Gut Microbiome data

```
 python3 pCNN_binary.py data/Malawiantwin_pairs
```
