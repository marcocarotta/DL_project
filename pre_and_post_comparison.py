# this script is needed to plot the loss
import matplotlib.pyplot as plt
import torch

def plot_train_loss_epochs():
        # open the checkpoint file and plot the loss
    checkpoint_path_pre = "models/validation/model_checkpoint_9_prelayernorm.pth" # for loading the checkpoint
    checkpoint_path_post = "models/validation/model_checkpoint_9_postlayernorm.pth" # for loading the checkpoint

    output_path = "loss_plots/comparison_pre_post.pdf" # for saving the figure

    checkpoint_pre = torch.load(checkpoint_path_pre,map_location=torch.device('cpu'))
    checkpoint_post = torch.load(checkpoint_path_post,map_location=torch.device('cpu'))

    print(checkpoint_pre.keys())
    print(checkpoint_post.keys())

    train_loss_pre = checkpoint_pre['train_loss']
    val_loss_pre = checkpoint_pre['val_loss']

    train_loss_post = checkpoint_post['train_loss']
    val_loss_post = checkpoint_post['val_loss']

    n_epochs_pre = checkpoint_pre['epoch']
    n_epochs_post = checkpoint_post['epoch']

    assert n_epochs_pre == n_epochs_post
    epochs = list(range(0, n_epochs_pre+1))

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss_pre, label='train loss pre layernorm',color='darkorange')
    plt.plot(epochs, val_loss_pre, label='val loss pre layernorm', color = 'darkorange', linestyle='dashed')
    plt.plot(epochs, train_loss_post, label='train loss post layernorm', color='steelblue')
    plt.plot(epochs, val_loss_post, label='val loss post layernorm', color='steelblue', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    print(f"Metrics saved to {output_path}")

def main():
    plot_train_loss_epochs()


if __name__ == '__main__':
    main()
