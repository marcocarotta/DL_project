# this script is needed to plot the loss
import matplotlib.pyplot as plt
import torch

def plot_train_loss_epochs():
        # open the checkpoint file and plot the loss
    checkpoint_path = "models/last_config/model_checkpoint_1.pth" # for loading the checkpoint
    tokens = checkpoint_path.split('/')
    output_path = "loss_plots/epoch_metrics_" + tokens[2] + ".pdf" # for saving the figure

    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    print(checkpoint.keys())

    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_accuracy = checkpoint['train_accuracy']
    val_accuracy = checkpoint['val_accuracy']

    n_epochs = checkpoint['epoch']
    epochs = list(range(0, n_epochs+1))

    # plt.figure(figsize=(12, 6))
    # plt.title(f"Metrics for {tokens[1:]}")
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_loss, label='train_loss')
    # plt.plot(epochs, val_loss, label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accuracy, label='train_accuracy')
    # plt.plot(epochs, val_accuracy, label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    
    # plt.savefig(output_path)

    print("Train Loss: ", train_loss,)
    print("Val Loss: ", val_loss)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    print(f"Metrics saved to {output_path}")

def plot_loss_step():
        # open the checkpoint file and plot the loss
    checkpoint_path = "models/last_config/model_checkpoint_1.pth" # for loading the checkpoint
    tokens = checkpoint_path.split('/')
    output_path = "loss_plots/step_metrics_" + tokens[2] + ".pdf" # for saving the figure

    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    print(checkpoint.keys())

    train_loss = checkpoint['step_train_loss']
    val_loss = checkpoint['step_val_loss']
    train_accuracy = checkpoint['train_accuracy']
    val_accuracy = checkpoint['val_accuracy']

    assert len(train_loss) == len(val_loss), "Train and validation loss have different lengths"
    n_steps = len(train_loss)
    steps = list(range(0, 100*n_steps, 100))
    
    # plt.figure(figsize=(12, 6))
    # plt.title(f"Metrics for {tokens[1:]}")
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_loss, label='train_loss')
    # plt.plot(epochs, val_loss, label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accuracy, label='train_accuracy')
    # plt.plot(epochs, val_accuracy, label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    
    # plt.savefig(output_path)

    print("Train Loss: ", train_loss,)
    print("Val Loss: ", val_loss)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, train_loss, label='train_loss')
    plt.plot(steps, val_loss, label='val_loss')
    plt.xlabel('Step (batch)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    print(f"Metrics saved to {output_path}")

def main():
    plot_loss_step()


if __name__ == '__main__':
    main()
