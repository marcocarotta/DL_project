# this script is needed to plot the loss
import matplotlib.pyplot as plt
import torch

def main():
    # open the checkpoint file and plot the loss
    checkpoint_path = "YOUR_PATH_HERE" # for loading the checkpoint
    output_path = "YOUR_PATH_HERE" # for saving the figure

    checkpoint = torch.load(checkpoint_path)
    train_loss = checkpoint['train_loss'] 
    val_loss = checkpoint['val_loss']
    train_accuracy = checkpoint['train_accuracy']
    val_accuracy = checkpoint['val_accuracy']

    n_epochs = checkpoint['epoch']
    epochs = list(range(1, n_epochs+1))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
