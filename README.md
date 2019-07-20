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
- seaborn >=0.9.0
- matplotlib >=3.1.0

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
usage: pCNN.py [-h] [--train] [--evaluation] [--test]
               [--data_dir <data_directory>] [--test_file TEST_FILE]
               [--correlation_file CORRELATION_FILE] [--model_dir MODEL_DIR]
               [--result_dir <data_directory>] [--outcome_type OUTCOME_TYPE]
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
  --train               Use this option for train model
  --evaluation          Use this option for evaluate model
  --test                Use this option for test model
  --data_dir <data_directory>
                        The data directory for training and evaluation
  --test_file TEST_FILE
                        The unlabelled test file
  --correlation_file CORRELATION_FILE
                        The correlation matrix for unlabelled test file
  --model_dir MODEL_DIR
                        The directory to save or restore the trained models.
  --result_dir <data_directory>
                        The directory to save test / evaluation result
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

### USA Human Gut Microbiome data (Continous-Outcome)
#### Train the model

The USA Human Gut Microbiome data contains 308 samples with 1087 OTUs. For details of description, please check our paper.
```
python3 src/pCNN.py --train --data_dir data/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
After training, the well-trained model will be saved to model directory.
#### Evaluate the well-trained model

```
python3 src/pCNN.py --evaluation --data_dir data/USA --result_dir result/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
The program will evaluate the well-trained model, draw a R-squared figure, and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/pCNN/blob/master/result/USA/result.jpg"/></div>
</center>  


#### Test the model with unlabelled data

```
python3 src/pCNN.py --test --test_file data/USA/X_test.npy  --correlation_file data/USA/c.npy --result_dir result/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
The program will take the unlabelled test file and save the prediction result to result directory.


### Malawian Twin pairs Human Gut Microbiome data (Binary-Outcome)
#### Train the model
The USA Human Gut Microbiome data contains 995 samples with 2291 OTUs.
```
python3 src/pCNN.py --train --data_dir data/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
#### Evaluate the well-trained model

```
python3 src/pCNN.py --evaluation --data_dir data/Malawiantwin_pairs --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
The program will draw a ROC figure and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/pCNN/blob/master/result/Malawiantwin_pairs/result.jpg"/></div>
</center>  

#### Test the model with unlabelled data
```
python3 src/pCNN.py --test --test_file data/Malawiantwin_pairs/X_test.npy --correlation_file data/Malawiantwin_pairs/c.npy --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
