# Introduction

This repository contains pytorch implementation of TransConv model for protein secondary structure prediction using a convolution infused transformer model.

# Getting Started 

Clone or download this repository and install the pre-requisites:

- pytorch
- transformers
- numpy
- matplotlib
- pandas
- sklearn

```
pip install torch transformers numpy matplotlib pandas scikit-learn
```


## Training model:
To train the proposed TransConv model from scratch use ```train.py```
```
python train.py --dataset_path attsec_dataset --dataset_type proteinnet
```
There are several other command line options supported by ```train.py``` some of which also dictate the hyperparameters of the model to train. Please go through the source code to understand the other arguments.

## Testing trained models:
```
python eval.py  --dataset_type proteinnet
```

The trained models are saved in the ```save_files``` directory.  
