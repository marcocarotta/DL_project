
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import logging
import math
from torch.utils.data import SubsetRandomSampler
import random


# Set up logging configuration
logging.basicConfig(level=logging.INFO) 

class Trainer:
    """
    Trainer class for training a PyTorch model.
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, lr=1e-4, scheduler_information=None, max_epochs=10, auxiliary_loss_percentage=0.5, multitask=False):
        """
        Initializes the Trainer class.
        
        Args:
            model (nn.Module): PyTorch model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            loss_fn (nn.Module): Loss function to use for training.
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
            lr (float): Learning rate.
            scheduler_information (dict): Dictionary containing scheduler type and parameters.
            max_epochs (int): Maximum number of epochs to train.
            auxiliary_loss_percentage (float): How much auxiliary loss impacts the total loss.
            multitask (bool): Whether the model is multitask.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.max_epochs = max_epochs
        self.auxiliary_loss_percentage = auxiliary_loss_percentage  # How much auxiliary loss impacts the total loss
        self.multitask = multitask
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self._initialize_scheduler(scheduler_information)

        # Track loss and accuracy for plotting (we will obtain also perplexity by exponentiating the loss)
        self.train_loss_tracking = [] 
        self.train_accuracy_tracking = [] 
        self.val_loss_tracking = [] 
        self.val_accuracy_tracking = [] 
        self.step_train_loss_tracking = []
        self.step_val_loss_tracking = []
        
        
        # Create data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)  

    def _initialize_scheduler(self, scheduler_info = None):
        """
        Initialize the learning rate scheduler.

        Args:
            scheduler_info (dict): Dictionary containing scheduler type and parameters.

        Example:
        For StepLR scheduler:
        scheduler_info = {
            "scheduler_type": "StepLR",
            "step_size": 5,
            "gamma": 0.5
        }
        For ReduceLROnPlateau scheduler:
        scheduler_info = {
            "scheduler_type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
        }
        """
        if scheduler_info is None:
            self.scheduler = None
            return
        if scheduler_info["scheduler_type"] == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_info["step_size"], gamma=scheduler_info["gamma"])
        elif scheduler_info["scheduler_type"] == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=scheduler_info["mode"], factor=scheduler_info["factor"], patience=scheduler_info["patience"])
        else:
            self.logger.warning("Scheduler type not recognized. No scheduler has been initialized during training, that is, learning rate will not be adjusted.")
            self.scheduler = None

    def _step_scheduler(self, val_loss):
        """
        Step the learning rate scheduler, if one is defined.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _compute_loss_and_accuracy(self, input, target, aux_logits=None):
        """
        Common function for computing loss and accuracy for both training and validation.
        """
        if self.multitask:
            logits, _ = self.model(input)  # Forward pass
        else:
            logits = self.model(input)
        
        # Compute main loss
        main_loss = self.loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))  # Main task loss
        total_loss = main_loss

        # If multitask, calculate auxiliary task loss and combine
        if self.multitask and aux_logits is not None:
            aux_loss = self.loss_fn(aux_logits.view(-1, aux_logits.size(-1)), target.view(-1))  # Auxiliary loss
            total_loss = (1 - self.auxiliary_loss_percentage) * main_loss + self.auxiliary_loss_percentage * aux_loss  # Weighted loss

        # Calculate accuracy
        _, predicted = logits.max(dim=-1)
        correct = (predicted == target).sum().item()
        total = target.numel()
        accuracy = correct / total

        return total_loss, accuracy

    def _run_epoch(self, mode="train"):
        """
        Common function to run training or validation for an epoch.
        """
        is_train = mode == "train"
        self.model.train() if is_train else self.model.eval()

        running_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        # Choose loader based on mode
        loader = self.train_loader if is_train else self.val_loader

        # Disable gradient calculation during validation
        with torch.set_grad_enabled(is_train):
            for idx, (input, target) in enumerate(loader):
                input, target = input.to(self.device), target.to(self.device)

                # Zero the gradients for training
                if is_train:
                    self.optimizer.zero_grad()

                # Compute loss and accuracy
                aux_logits = None  # You can modify this for multitask purposes
                total_loss, accuracy = self._compute_loss_and_accuracy(input, target, aux_logits)

                # Backpropagation for training
                if is_train:
                    total_loss.backward()
                    self.optimizer.step()

                running_loss += total_loss.item()
                total_accuracy += accuracy
                total_samples += 1

                # tracking step by step loss, i will not be able to plot this to on the same graph
                # as they will have different length
                if idx % 100 == 0:  # Every 100 batches, log progress
                    estimate_loss_for_logging = self.estimate_loss()
                    self.step_train_loss_tracking.append(estimate_loss_for_logging['train_loss'])
                    self.step_val_loss_tracking.append(estimate_loss_for_logging['val_loss'])

                    self.logger.info(f"Epoch progress [{idx+1}/{len(loader)}],Train Loss: {estimate_loss_for_logging['train_loss']:.4f}, Train Perplexity: {estimate_loss_for_logging['train_perplexity']:.4f}, Val Loss: {estimate_loss_for_logging['val_loss']:.4f}, Val Perplexity: {estimate_loss_for_logging['val_perplexity']:.4f}")
        avg_loss = running_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        return avg_loss, avg_accuracy


    @torch.no_grad()
    def estimate_loss(self, eval_iters=50):
        """
        Estimate loss and perplexity on a random subset of the dataset. 
        It averages the loss and perplexity over the subset.

        Args:
            eval_iters (int): Number of iterations to evaluate on.

        Returns:
            dict: Dictionary containing estimated loss and perplexity for training and validation sets.
        """
        out = {}
        self.model.eval()
    
        for split in ['train', 'val']:
            running_loss = 0.0
            total_samples = 0
    
            # Choose the appropriate loader
            loader = self.train_loader if split == 'train' else self.val_loader
            
            # Sample a subset of the dataset
            sampler = SubsetRandomSampler(
                random.sample(range(len(loader.dataset)), k=min(eval_iters, len(loader.dataset)))
            )
            subset_loader = torch.utils.data.DataLoader(
                loader.dataset, batch_size=loader.batch_size, sampler=sampler, num_workers=4
            )
    
            for input, target in subset_loader:
                input, target = input.to(self.device), target.to(self.device)
                aux_logits = None
                total_loss, _ = self._compute_loss_and_accuracy(input, target, aux_logits)
                running_loss += total_loss.item() * input.size(0)  # Account for batch size
                total_samples += input.size(0)
    
            avg_loss = running_loss / total_samples
            out[f'{split}_loss'] = avg_loss
            out[f'{split}_perplexity'] = math.exp(avg_loss)
    
        self.model.train()        
        return out

    def train(self):
        # Training loop
        for epoch in range(self.max_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} - Training...")
            train_loss, train_accuracy = self._run_epoch(mode="train")
            self.logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f},Perplexity {math.exp(train_loss):.4f}, Train Accuracy: {train_accuracy*100:.2f}%")

            self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} - Validating...")
            val_loss, val_accuracy = self._run_epoch(mode="val")
            self.logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
            
            # Save loss and accuracy for plotting
            self.train_loss_tracking.append(train_loss)
            self.train_accuracy_tracking.append(train_accuracy)
            self.val_loss_tracking.append(val_loss)
            self.val_accuracy_tracking.append(val_accuracy)

            # Step the scheduler based on validation loss
            self._step_scheduler(val_loss)  # If scheduler is None, this will be a no-op

            # Optionally save checkpoint after every epoch
            self.save_checkpoint(epoch, filename=f"model_checkpoint_{epoch}.pth")

        self.logger.info("Training complete.")

    def save_checkpoint(self, epoch, filename="model_checkpoint.pth"):
        # Save model and optimizer state to cpu to avoid device mismatch during loading
        self.model.to("cpu")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss_tracking,
            'train_accuracy': self.train_accuracy_tracking,
            'val_loss': self.val_loss_tracking,
            'val_accuracy': self.val_accuracy_tracking,
            'step_train_loss':self.step_train_loss_tracking,
            'step_val_loss':self.step_val_loss_tracking
        }, filename)

        self.model.to(self.device)
        self.logger.info(f"Checkpoint saved to {filename}")
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="model_checkpoint.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Checkpoint loaded from {filename}")

        # Load loss and accuracy tracking
        self.train_loss_tracking = checkpoint['train_loss']
        self.train_accuracy_tracking = checkpoint['train_accuracy']
        self.val_loss_tracking = checkpoint['val_loss']
        self.val_accuracy_tracking = checkpoint['val_accuracy']

if __name__ == "__main__":
    from dataset import CharDataset
    from model import CharTransformer

    with open ('data/dataset.txt', 'r') as f:
        data = f.read()

    dataset = CharDataset(data, block_size=128).preprocess()
    # Split the data into train and validation sets

    train_size = 0.8
    overlap = 0
    train_dataset, val_dataset = dataset.train_val_split(train_size, overlap)

    # Example instantiation
    vocab_size = dataset.vocabulary_size  # Example vocabulary size for the Shakespeare dataset
    model = CharTransformer(vocab_size, embed_dim=32, num_heads=1, num_layers=1, ff_hid_dim= 64, block_size=128)
    model.summary()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    scheduler_info = {
        "scheduler_type": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.1,
        "patience": 10,
    }
    max_epochs = 1
    auxiliary_loss_percentage = 0.5
    multitask = False

    # Instantiate the trainer
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, scheduler_information=scheduler_info, max_epochs=max_epochs, auxiliary_loss_percentage=auxiliary_loss_percentage, multitask=multitask)

    # Train the model
    trainer.train()
