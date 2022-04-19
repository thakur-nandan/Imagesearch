import argparse
import logging
import sys
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from imagesearch import LoggingHandler
from imagesearch.dataset.dataset import CIFAR_LABELS
from imagesearch.db import ImageDatabase
from imagesearch.models import load_model
from imagesearch.dataset import load_cifar10

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", type=str)
    parser.add_argument("--distance", dest="search_by_distance", action='store_true')
    parser.add_argument("--score", dest="search_by_score", action='store_true')
    parser.add_argument("--label", dest="label", type=int, default=0)
    parser.add_argument("--index", dest="index", type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--evaluate-samples", dest="evaluate_samples", type=int, default=0)
    parser.add_argument("--output", dest="output", type=str, default="search-results.png")
    args = vars(parser.parse_args(sys.argv[1:]))
    model_path = args['model_path']
    k = args['k']

    logging.info("loading CIFAR10 dataset...")
    train, test = load_cifar10()
    device = get_device()
    
    if model_path:
        logging.info("loading model from {}...".format(os.path.abspath(model_path)))
        net = load_model(model_path=model_path, device=device)
        logging.info("loaded model...")

    logging.info("loading database...")
    net = net.to(torch.device('cpu'))
    db = ImageDatabase(train, net, torch.device('cpu'))
    embs = db.index.detach().cpu().numpy()
    lbls = db.labels.detach().cpu().numpy()

    # fig = plt.figure(figsize=(11, 8), dpi=300)
    # ax = plt.axes(projection="3d")
    # for cls in range(10):
    #     logging.info("running T-SNE for label {}".format(CIFAR_LABELS[cls]))
    #     pts = embs[np.argwhere(lbls == cls).flatten()]
    #     if pts.shape[0] < 2:
    #         continue
    #     tsne = TSNE(n_components=3).fit_transform(pts)
    #     logging.info("done T-SNE")
    #     ax.plot3D(tsne[:, 0], tsne[:, 1], tsne[:, 2], '.', label=CIFAR_LABELS[cls])
    # plt.legend()
    # plt.savefig(args['output'])

    fig = plt.figure(figsize=(11, 8), dpi=300)
    for cls in range(10):
        logging.info("running T-SNE for label {}".format(CIFAR_LABELS[cls]))
        pts = embs[np.argwhere(lbls == cls).flatten()]
        if pts.shape[0] < 2:
            continue
        tsne = TSNE(n_components=2).fit_transform(pts)
        logging.info("done T-SNE")
        plt.plot(tsne[:, 0], tsne[:, 1], '.', label=CIFAR_LABELS[cls])
    plt.legend()
    plt.savefig(args['output'])