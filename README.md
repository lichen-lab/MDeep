## Phylogeny-regularized Convolutional NeuralNetwork
We develop a deep learning prediction method "Phylogeny-regularized convolutional Neural Network,"(pCNN) for microbiome-based prediction. The advantage of pCNN is that it uses the convolutional kernel to capture the signals of microbiome species with close evolutionary relationship in a local receptive field. Moreover, pCNN uses different convolutional layer to capture different taxonomic rank (e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced by the phylogenetic tree.

<center>

<div align=center><img width="600" height="400" src="https://raw.githubusercontent.com/alfredyewang/pCNN/master/docs/Architecture.jpg"/></div>
</center>  

For details of pCNN, users can refer to our paper "**A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based Predictionsn**".


## Requirements and Installation

pCNN is implemented by TensorFlow. Both CPU and GPU mode are supported. Please check the guide on official website for detail instruction of installing TensorFlow-GPU.

- Python 3.6
- TensorFlow == 1.12.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3

Download pCNN:
```
git clone https://github.com/alfredyewang/pCNN
```
Install requirements
```
pip3 install -r requirements.txt
```
## Usage
You can see the hyper paramenters for pCNN by help option:

```
usage: pCNN.py [-h] [--data_dir <data_directory>] [--model_dir MODEL_DIR]
               [--train] [--test] [--outcome_type OUTCOME_TYPE]
               [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH]
               [--learning_rate LEARNING_RATE] [--dropout_rate DROPOUT_RATE]
               [--L2_regularizer L2_REGULARIZER]
               [--window_size WINDOW_SIZE [WINDOW_SIZE ...]]
               [--kernel_size KERNEL_SIZE [KERNEL_SIZE ...]]
               [--strides STRIDES [STRIDES ...]]

A Phylogeny-regularized Convolutional NeuralNetwork for Microbiome-based
Predictions

optional arguments:
  -h, --help            show this help message and exit
  --data_dir <data_directory>
                        The data directory
  --model_dir MODEL_DIR
                        The directory to save or restore the trained models.
  --train               Use this option for training model
  --test                Use this option for testing model
  --outcome_type OUTCOME_TYPE
                        The outcome type
  --batch_size BATCH_SIZE
                        The batch size for training
  --max_epoch MAX_EPOCH
                        The max epoch for training
  --learning_rate LEARNING_RATE
                        The learning rate for training
  --dropout_rate DROPOUT_RATE
                        The dropout rate for training
  --L2_regularizer L2_REGULARIZER
                        The L2 lambda
  --window_size WINDOW_SIZE [WINDOW_SIZE ...]
                        The window size for convolutional layers
  --kernel_size KERNEL_SIZE [KERNEL_SIZE ...]
                        The kernel size for convolutional layers
  --strides STRIDES [STRIDES ...]
                        The strides size for convolutional layers

```

## Example

#### Train the model based on USA Human Gut Microbiome data (Continous-Outcome)
The USA Human Gut Microbiome data contains 308 samples with 1087 OTUs. For details of description , users can refer to our paper
```
python3 src/pCNN.py --data_dir data/USA --train --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
#### Test the model

```
python3 src/pCNN.py --data_dir data/USA --test --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```

### Train the model based on Malawiantwin pairs Human Gut Microbiome data (Binary-Outcome)
The USA Human Gut Microbiome data contains 995 samples with 2291 OTUs.
```
python3 src/pCNN.py --data_dir data/Malawiantwin_pairs --train --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
#### Test the model
```
python3 src/pCNN.py --data_dir data/Malawiantwin_pairs --test --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
