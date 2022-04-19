import argparse
import logging
import matplotlib.pyplot as plt
import sys
import os
import torch

from imagesearch import LoggingHandler
from imagesearch.db import ImageDatabase
from imagesearch.models import UvaEncoder, load_model_uva
from imagesearch.models import ImageEncoder, load_model
from imagesearch.dataset import CIFAR_LABELS, load_cifar10, TripletDataset
from training.train_loop import train

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
    db = ImageDatabase(test, net, device)
    score_recall, score_map, score_acc = db.evaluate_all(test, top_k=10)