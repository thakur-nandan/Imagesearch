'''
Main training loop

'''
import argparse
from re import I
import torch
import os
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

from imagesearch import LoggingHandler
from imagesearch.dataset import download_cifar10, load_cifar10, TripletDataset, RandomSubsetSampler
from imagesearch.models import ImageEncoder, ImageDecoder

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_ds, test_ds, n_samples, n_epochs, model_path=None, device=None, output_vector_size=10, lambda_val=0.7, loss_plot_path=None):
    if device is None:
        device = get_device()
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=RandomSubsetSampler(len(train_ds), n_samples)
    )
    n_test_samples = max(1, round(n_samples * 0.3))
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        sampler=RandomSubsetSampler(len(test_ds), n_test_samples)
    )
    enc = ImageEncoder(output_vector_size=output_vector_size).to(device)
    dec = ImageDecoder(input_vector_size=output_vector_size).to(device)
    best_enc = enc
    enc_optimizer = torch.optim.Adam(enc.parameters())
    dec_optimizer = torch.optim.Adam(dec.parameters())
    mse_loss = torch.nn.MSELoss()
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y))
    # loss_fn = torch.nn.TripletMarginLoss()
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(1,n_epochs+1):
        total_loss = 0
        n_batches = 0
        test_loss = 0
        n_test_batches = 0

        for data in train_loader:
            a, p, n = data

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            z_a = enc(a)
            z_p = enc(p)
            z_n = enc(n)

            loss = lambda_val * loss_fn(z_a, z_p, z_n)
            loss += mse_loss(a, dec(z_a))
            loss.backward()

            enc_optimizer.step()
            dec_optimizer.step()
            total_loss += loss
            n_batches += 1

        for data in test_loader:
            with torch.no_grad():
                a, p, n = data
                z_a = enc(a)
                z_p = enc(p)
                z_n = enc(n)

                loss = lambda_val * loss_fn(z_a, z_p, z_n)
                loss += mse_loss(a, dec(z_a))
                test_loss += loss
                n_test_batches += 1

        loss = total_loss / n_batches
        test_loss = test_loss / n_test_batches
        train_losses.append(loss)
        test_losses.append(test_loss)
        logging.info("epoch: {} loss: {:.4f} test-loss: {:.4f}".format(epoch, loss, test_loss))

        if test_loss < best_loss:
            if model_path:
                logging.info("saving model to {}".format(model_path))
                # model_dir = os.path.dirname(model_path)
                # try:
                #     os.makedirs(model_dir)
                # except Exception as e:
                #     logging.warn("Error trying to make model output directory: {}".format(str(e)))
                #     continue
                torch.save({
                    "epoch": epoch,
                    "loss": test_loss,
                    "model": enc.state_dict(),
                    "output_vector_size":enc.get_output_vector_size()
                }, model_path)
            best_loss = test_loss
            best_enc = ImageEncoder(output_vector_size=output_vector_size)
            best_enc.load_state_dict(enc.state_dict())

    if not loss_plot_path is None:
        plt.figure(figsize=(11,8), dpi=300)
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_plot_path)

    return best_enc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', dest='samples', type=int, default=50000)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--lambda', dest='lambda_val', type=float, default=0.7)
    parser.add_argument('--vector-size', dest='output_vector_size', type=int, default=10)
    parser.add_argument('--output', dest='model_path', type=str)
    parser.add_argument('--loss-plot', dest='loss_plot_path', type=str)
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

    train_ds = TripletDataset(train_dic, device=device)
    test_ds = TripletDataset(test_dic, device=device)

    n_samples = args['samples']
    n_epochs = args['epochs']  # total number of epochs
    model_path = args['model_path']
    output_vector_size = args['output_vector_size']
    lambda_val = args['lambda_val']
    loss_plot_path = args['loss_plot_path']

    logging.info("training. epochs={} samples={}".format(n_epochs, n_samples))
    train(train_ds, test_ds, n_samples, n_epochs, model_path, output_vector_size=output_vector_size, lambda_val=lambda_val, loss_plot_path=loss_plot_path)
    logging.info("done training")
