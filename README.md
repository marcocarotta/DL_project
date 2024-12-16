# Deep Learning Project

This repository contains the project for the Deep Learning course.

The aim of the project is to implement a transformer-based, character-level language model (GPT-like) and train it on the Shakespeare dataset.

## Pipeline
The pipeline is the following:
```bash 
project/
│
├── dataset.py          # Data loading and preprocessing.
├── model.py            # Transformer language model definition.
├── train.py            # Training and validation logic.
├── generate.py         # Text generation logic.
├── main.ipynb          # Pipeline orchestration (entry point). [Using .ipynb for train on Google Colab or Kaggle]
│
└── data/               # Store raw and/or preprocessed datasets.
```
In the following we give more information about each component of the pipeline

## dataset.py
The dataset is handled by the class `CharDataset` that extends the PyTorch class `Dataset`. The principal methods of this class are the `__getitem__` and the `get_stoi`/`get_itos`.

- `__getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]`: Takes an index and returns two torch tensors, respectively the input sequence and the output sequence.
- `preprocess(self, lowercase: bool = True, punctuation: bool = False)`: Preprocess the text and update the dataset.
- `train_val_split(self, train_size: float = 0.8, overlap: int = 250)`: Split the dataset into training and validation sets.
- `update_dataset(self, text: str)`: Update the dataset with new text.
- `dataset_analysis(self)`: Comprehensive analysis of the current dataset state.
- `remove_less_frequent_chars(self, threshold: int = 1)`: Remove characters with frequency below a given threshold and substitute them with a UNK token.
- `get_stoi(self) -> dict`: Returns the string-to-index mapping.
- `get_itos(self) -> dict`: Returns the index-to-string mapping.
## model.py
The model is defined in the `model.py` file and consists of three main classes:

- `CausalSelfAttention`: Implements a multi-head causal self-attention mechanism.
- `TransformerBlock`: Implements a single transformer block with causal self-attention and feed-forward network.
- `CharTransformer`: Implements a character-level transformer model.

The `CharTransformer` class includes methods for forward pass and model summary.

## train.py
The training logic is implemented in the `Trainer` class. Key methods include:

- `__init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, lr=1e-4, scheduler_information=None, max_epochs=10, auxiliary_loss_percentage=0.5, multitask=False)`: Initializes the `Trainer` class.
- `_initialize_scheduler(self, scheduler_info=None)`: Initialize the learning rate scheduler.
- `_step_scheduler(self, val_loss)`: Step the learning rate scheduler, if one is defined.
- `_compute_loss_and_accuracy(self, input, target, aux_logits=None)`: Common function for computing loss and accuracy for both training and validation.
- `_run_epoch(self, mode="train")`: Common function to run training or validation for an epoch.
- `estimate_loss(self, eval_iters=50)`: Estimate loss and perplexity on a random subset of the dataset.
- `train(self)`: Training loop.
- `save_checkpoint(self, epoch, filename="model_checkpoint.pth")`: Save model and optimizer state to a checkpoint.
- `load_checkpoint(self, filename="model_checkpoint.pth")`: Load model and optimizer state from a checkpoint.

## generate.py
The text generation logic is implemented in the `TextGenerator` class. Key methods include:

- `__init__(self, model, dataset, block_size=100, device='cpu')`: Initializes the `TextGenerator` class.
- `generate(self, start_text, length=100, temperature=1.0)`: Generate text starting from the given `start_text`.
- `get_next_char(self, probs, p=0.9)`: Implement top-p (nucleus) sampling.
- `load_model(self, model_path)`: Load the model from a checkpoint.
