This repository contains the project for the Deep learning course.

The aim of the project is to implement a transformer-based, character-level language model (GPT-like) and train it on the Shakespeare dataset.

# Pipeline
The pipeline is the following:
```bash 
project/
│
├── dataset.py          # Data loading and preprocessing.
├── model.py            # Transformer language model definition.
├── train.py            # Training and validation logic.
├── generate.py         # Text generation logic.
├── main.py             # Pipeline orchestration (entry point). [NOT IMPLEMENTED YET]
├── main.ipynb          # Pipeline orchestration (entry point). [Using .ipynb for train on Google Colab]
│
└── data/               # Store raw and/or preprocessed datasets.
```
In the following we give more information about each component of the pipeline
## dataset.py
The dataset is handled by the class `CharDataset` that extend the PyTorch class `Dataset`.
The principal methods of this class are the `__getitem__` and the `get_stoi`/`get_itos`. 

The method `__getitem__` takes an index and returns two torch tensor, respectively the input sequence and the output sequence.

TODO need to solve the problem with train and validation dataset. maybe do something like a method from the dataset made with all the text return different CharDataset with consistent stoi itos but different text samples.

## model.py

TODO create a google drive folder with model as they are too big for git.

## train.py

## generate.py





ideas
where to put layer norm
pre post layer norm
