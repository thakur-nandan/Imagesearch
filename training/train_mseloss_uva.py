'''
Main training loop

'''
import argparse
import torch
import os
import logging
from torch.utils.data import DataLoader
import sys

from imagesearch import LoggingHandler
from imagesearch.dataset import download_cifar10, load_cifar10, AutoDataset
from imagesearch.models import UvaEncoder, UvaDecoder

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_ds, test_ds, n_epochs, model_path=None, device=None, batch_size=64, lr=0.001, latent_dim=64):
    if device is None:
        device = get_device()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    encoder = UvaEncoder(latent_dim=latent_dim).to(device)
    decoder = UvaDecoder(latent_dim=latent_dim).to(device)
    
    # Both the enocder and decoder parameters
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())

    # Adam Optimizer
    logging.info("Using Adam optimizer with learning rate = 0.001")
    optimizer = torch.optim.Adam(autoencoder_params, lr=lr)
    
    # Loss function (MSE Loss)
    logging.info("Using MSE loss function")
    loss_fn = torch.nn.MSELoss()
    
    best_loss = float('inf')

    logging.info("Starting to train for {} epochs".format(n_epochs))
    for epoch in range(1, n_epochs+1):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        test_loss = test_step(encoder, decoder, test_loader, loss_fn, device=device)
        logging.info("Epoch: {} Training loss: {:.5f} Test loss: {:.5f}".format(epoch, train_loss, test_loss))

        if test_loss < best_loss:
            if model_path:
                logging.info("saving model to {}".format(model_path))
                os.makedirs(model_path, exist_ok=True)
                torch.save(encoder.state_dict(), os.path.join(model_path, "uva_encoder_{}_model.pt".format(latent_dim)))
                torch.save(decoder.state_dict(), os.path.join(model_path, "uva_decoder_{}_model.pt".format(latent_dim)))
            
            best_loss = test_loss

    return encoder, decoder


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model UvaEncoder
    decoder: A convolutional Decoder. E.g. torch_model UvaDecoder
    train_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    optimizer: PyTorch optimizer.
    device: "cuda" or "cpu"
    Returns: Train Loss
    """
    #  Set networks to train mode.
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        # Move images to device
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        
        # Zero grad the optimizer
        optimizer.zero_grad()
        # Feed the train images to encoder
        enc_output = encoder(train_img)
        # The output of encoder is input to decoder !
        dec_output = decoder(enc_output)
        
        # Decoder output is reconstructed image
        # Compute loss with it and orginal image which is target image.
        loss = loss_fn(dec_output, target_img)
        # Backpropogate
        loss.backward()
        # Apply the optimizer to network by calling step.
        optimizer.step()
    # Return the loss
    return loss.item()

def test_step(encoder, decoder, val_loader, loss_fn, device):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model UvaEncoder
    decoder: A convolutional Decoder. E.g. torch_model UvaDecoder
    val_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    device: "cuda" or "cpu"
    Returns: Validation Loss
    """
    
    # Set to eval mode.
    encoder.eval()
    decoder.eval()
    
    # We don't need to compute gradients while validating.
    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            # Move to device
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            # Again as train. Feed encoder the train image.
            enc_output = encoder(train_img)
            # Decoder takes encoder output and reconstructs the image.
            dec_output = decoder(enc_output)

            # Validation loss for encoder and decoder.
            loss = loss_fn(dec_output, target_img)
    # Return the loss
    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--output', dest='model_path', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', dest='latent_dim', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4)
    args = vars(parser.parse_args(sys.argv[1:]))

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    device = get_device()
    logging.info("Device used: {}".format(device))

    #### Download CIFAR10 dataset ####
    dataset_dir = "./datasets/cifar-10-batches-py"
    if not os.path.isdir(dataset_dir):
        download_cifar10()

    #### Load CIFAR10 dataset ####
    train_dic, test_dic = load_cifar10()

    train_ds = AutoDataset(train_dic, device=device)
    test_ds = AutoDataset(test_dic, device=device)

    n_epochs = args['epochs']  # total number of epochs
    model_path = args['model_path']
    batch_size = args['batch_size']
    lr = args['lr']
    latent_dim = args["latent_dim"]

    logging.info("Starting to Train for epochs={}".format(n_epochs))
    train(train_ds=train_ds, test_ds=test_ds, n_epochs=n_epochs, model_path=model_path, batch_size=batch_size, lr=lr, latent_dim=latent_dim)
    logging.info("Finished training...")
